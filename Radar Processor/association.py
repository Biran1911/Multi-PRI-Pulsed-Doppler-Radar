# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:48:34 2026

@author: royi.b
Association module for target tracking and range unfolding.
"""
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Target:
    """Target track structure"""
    id: int
    filt_rng: float  # Filtered range (m)
    filt_vel: float  # Filtered velocity (m/s)
    power_dB: float  # Power (dB)
    doppler_bin: int
    range_bin: int
    pri_mask: int  # Bitmask of which PRIs detected this target
    hits: int  # Total number of associations
    misses: int  # Consecutive misses
    age: int  # Age in cycles
    last_cycle: int  # Last cycle this target was updated
    status: str = 'tentative'  # 'tentative' or 'confirmed'
    
    # Additional state for filtering
    pred_rng: float = 0.0
    pred_vel: float = 0.0


class TargetManager:
    """Manages collection of targets with unique ID assignment"""
    def __init__(self):
        self.targets: List[Target] = []
        self.next_id: int = 1
        
    def add_target(self, target: Target) -> None:
        """Add a new target to the manager"""
        target.id = self.next_id
        self.next_id += 1
        self.targets.append(target)
        
    def remove_target(self, target: Target) -> None:
        """Remove a target from the manager"""
        self.targets.remove(target)
        
    def __iter__(self):
        return iter(self.targets)
    
    def __len__(self):
        return len(self.targets)
    
    def get_by_id(self, target_id: int) -> Target:
        """Get target by ID"""
        for t in self.targets:
            if t.id == target_id:
                return t
        return None


def unfold_targets_from_plot_manager(
    plots: List[Dict],
    fc: float,
    r_tol: float = 50.0,
    v_tol: float = 2.0,
    min_pri_support: int = 1
) -> List[Dict]:
    """
    Unfold ambiguous plots and cluster them into potential targets.
    
    Args:
        plots: List of plot dictionaries with keys:
               ['PRI_us', 'doppler_bin', 'range_bin', 'power_dB', 
                'doppler_freq', 'amb_vel', 'amb_rng', 'CFAR_threshold_dB', 'CUT_dB']
        fc: Carrier frequency (Hz)
        r_tol: Range tolerance for clustering (m)
        v_tol: Velocity tolerance for clustering (m/s)
        min_pri_support: Minimum number of PRIs that must detect the target
    
    Returns:
        List of unfolded target dictionaries
    """
    if not plots:
        return []
    
    # Group plots by range and velocity proximity
    clusters = []
    
    for plot in plots:
        # Extract scalar values from potential numpy arrays
        pri_val = plot['PRI_us']
        pri_idx = int(pri_val.item()) if hasattr(pri_val, 'item') else int(pri_val)
        
        rng_val = plot['amb_rng']
        amb_rng = float(rng_val.item()) if hasattr(rng_val, 'item') else float(rng_val)
        
        vel_val = plot['amb_vel']
        amb_vel = float(vel_val.item()) if hasattr(vel_val, 'item') else float(vel_val)
        
        # Find matching cluster
        matched = False
        for cluster in clusters:
            # Check if plot is close to cluster centroid
            r_diff = abs(amb_rng - cluster['mean_rng'])
            v_diff = abs(amb_vel - cluster['mean_vel'])
            
            if r_diff < r_tol and v_diff < v_tol:
                # Add to cluster
                cluster['plots'].append(plot)
                # Update centroid
                n = len(cluster['plots'])
                cluster['mean_rng'] = (cluster['mean_rng'] * (n-1) + amb_rng) / n
                cluster['mean_vel'] = (cluster['mean_vel'] * (n-1) + amb_vel) / n
                cluster['pri_mask'] |= int(1 << pri_idx)
                matched = True
                break
        
        if not matched:
            # Create new cluster
            clusters.append({
                'plots': [plot],
                'mean_rng': amb_rng,
                'mean_vel': amb_vel,
                'pri_mask': int(1 << pri_idx)
            })
    
    # Convert clusters to target format
    targets = []
    for cluster in clusters:
        # Check PRI support
        pri_count = bin(cluster['pri_mask']).count('1')
        if pri_count < min_pri_support:
            continue
        
        # Average measurements - ensure scalars
        n_plots = len(cluster['plots'])
        
        avg_rng = 0.0
        avg_vel = 0.0
        avg_power = 0.0
        
        for p in cluster['plots']:
            rng_val = p['amb_rng']
            avg_rng += float(rng_val.item()) if hasattr(rng_val, 'item') else float(rng_val)
            
            vel_val = p['amb_vel']
            avg_vel += float(vel_val.item()) if hasattr(vel_val, 'item') else float(vel_val)
            
            pwr_val = p['power_dB']
            avg_power += float(pwr_val.item()) if hasattr(pwr_val, 'item') else float(pwr_val)
        
        avg_rng /= n_plots
        avg_vel /= n_plots
        avg_power /= n_plots
        
        # Use first plot for discrete values
        first_plot = cluster['plots'][0]
        
        dop_val = first_plot['doppler_bin']
        doppler_bin = int(dop_val.item()) if hasattr(dop_val, 'item') else int(dop_val)
        
        rng_bin_val = first_plot['range_bin']
        range_bin = int(rng_bin_val.item()) if hasattr(rng_bin_val, 'item') else int(rng_bin_val)
        
        targets.append({
            'filt_rng': avg_rng,
            'filt_vel': avg_vel,
            'power_dB': avg_power,
            'doppler_bin': doppler_bin,
            'range_bin': range_bin,
            'pri_mask': cluster['pri_mask'],
            'n_plots': n_plots
        })
    
    return targets


def update_target_manager(
    target_manager: TargetManager,
    new_targets: List[Dict],
    cycle_idx: int,
    r_gate: float = 100.0,
    v_gate: float = 5.0,
    confirm_threshold: int = 3,
    delete_threshold: int = 5,
    alpha: float = 0.7,
    beta: float = 0.3
) -> None:
    """
    Associate new targets with existing tracks and update target manager.
    
    Args:
        target_manager: TargetManager instance
        new_targets: List of new target detections from unfold_targets_from_plot_manager
        cycle_idx: Current cycle index
        r_gate: Range gate for association (m)
        v_gate: Velocity gate for association (m/s)
        confirm_threshold: Number of hits to confirm a track
        delete_threshold: Number of consecutive misses before deletion
        alpha: Position smoothing factor (0-1)
        beta: Velocity smoothing factor (0-1)
    """
    # Age all existing targets and increment misses
    for target in target_manager:
        target.age += 1
        target.misses += 1
    
    # Create association matrix
    associated_targets = set()
    associated_detections = set()
    
    # Nearest-neighbor association with gating
    for det_idx, detection in enumerate(new_targets):
        best_match = None
        best_distance = float('inf')
        
        for trk_idx, track in enumerate(target_manager):
            # Check if track is already associated
            if trk_idx in associated_targets:
                continue
            
            # Calculate gated distance
            r_diff = abs(detection['filt_rng'] - track.filt_rng)
            v_diff = abs(detection['filt_vel'] - track.filt_vel)
            
            # Gate check
            if r_diff > r_gate or v_diff > v_gate:
                continue
            
            # Normalized Mahalanobis-like distance
            distance = np.sqrt((r_diff / r_gate)**2 + (v_diff / v_gate)**2)
            
            if distance < best_distance:
                best_distance = distance
                best_match = trk_idx
        
        # Associate if match found
        if best_match is not None:
            track = target_manager.targets[best_match]
            
            # Update track with alpha-beta filter
            track.filt_rng = alpha * detection['filt_rng'] + (1 - alpha) * track.filt_rng
            track.filt_vel = alpha * detection['filt_vel'] + (1 - alpha) * track.filt_vel
            track.power_dB = alpha * detection['power_dB'] + (1 - alpha) * track.power_dB
            track.doppler_bin = detection['doppler_bin']
            track.range_bin = detection['range_bin']
            track.pri_mask = detection['pri_mask']
            
            # Update statistics
            track.hits += 1
            track.misses = 0
            track.last_cycle = cycle_idx
            
            # Check for confirmation
            if track.hits >= confirm_threshold:
                track.status = 'confirmed'
            
            associated_targets.add(best_match)
            associated_detections.add(det_idx)
    
    # Create new tracks for unassociated detections
    for det_idx, detection in enumerate(new_targets):
        if det_idx not in associated_detections:
            new_track = Target(
                id=0,  # Will be assigned by manager
                filt_rng=detection['filt_rng'],
                filt_vel=detection['filt_vel'],
                power_dB=detection['power_dB'],
                doppler_bin=detection['doppler_bin'],
                range_bin=detection['range_bin'],
                pri_mask=detection['pri_mask'],
                hits=1,
                misses=0,
                age=0,
                last_cycle=cycle_idx,
                status='tentative'
            )
            target_manager.add_target(new_track)
    
    # Remove tracks that have too many misses
    tracks_to_remove = []
    for track in target_manager:
        if track.misses >= delete_threshold:
            tracks_to_remove.append(track)
    
    for track in tracks_to_remove:
        target_manager.remove_target(track)


def predict_targets(target_manager: TargetManager, dt: float = 0.1) -> None:
    """
    Predict target positions for next cycle (optional, for constant velocity model).
    
    Args:
        target_manager: TargetManager instance
        dt: Time delta between cycles (seconds)
    """
    for target in target_manager:
        target.pred_rng = target.filt_rng + target.filt_vel * dt
        target.pred_vel = target.filt_vel