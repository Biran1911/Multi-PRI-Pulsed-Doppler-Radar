# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:48:53 2026

@author: royi.b
Association module for target tracking and range unfolding.

Visualization module for plotting Range-Doppler maps and detections.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_rd_maps(RD_maps, DR, detections, pre_candidates):
    """
    Plot Range-Doppler maps for each PRI and overlay CFAR detections.
    
    Parameters
    ----------
    RD_maps : dict or ndarray
        Dictionary of {PRI_us: {'fft_db': ndarray, ...}} for multiple PRIs,
        or a single 2D ndarray for a single RD map
    DR : float
        Dynamic range (dB) for color scale
    detections : list
        List of confirmed detections (from CFAR)
    pre_candidates : list
        List of pre-detection candidates (optional)
    """
    # Handle single RD map case
    if isinstance(RD_maps, np.ndarray):
        # Convert single map to dictionary format
        RD_maps = {'Single': {'fft_db': RD_maps}}
    
    num_maps = len(RD_maps)
    
    # Create figure with appropriate layout
    if num_maps == 1:
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        axs = [axs]  # Make it iterable
    else:
        fig, axs = plt.subplots(1, num_maps, figsize=(17, 8), sharey=True)
        if num_maps == 1:
            axs = [axs]
    
    # Group detections by PRI for fast lookup
    det_by_pri = {}
    if detections:
        for d in detections:
            pri_key = d.get('PRI_us', 'Single')
            det_by_pri.setdefault(pri_key, []).append(d)
    
    pre_by_pri = {}
    if pre_candidates:
        for p in pre_candidates:
            pri_key = p.get('PRI_us', 'Single')
            pre_by_pri.setdefault(pri_key, []).append(p)
    
    for i, (pri_us, data) in enumerate(sorted(RD_maps.items())):
        ax = axs[i]
        
        # Handle both dict and direct array formats
        if isinstance(data, dict):
            fft_db = data['fft_db']
        else:
            fft_db = data
            
        num_doppler_bins, num_range_bins = fft_db.shape
        
        # Doppler bin axis centered around 0
        doppler_bins = np.arange(-num_doppler_bins // 2, num_doppler_bins // 2)
        range_bins = np.arange(num_range_bins)
        
        ax.imshow(
            fft_db.T,
            aspect='auto',
            cmap='jet',
            origin='lower',
            vmin=np.max(fft_db) - DR,
            vmax=np.max(fft_db),
            extent=[doppler_bins[0], doppler_bins[-1],
                    range_bins[0], range_bins[-1]]
        )
        
        # Plot pre-detection candidates (if any)
        if pri_us in pre_by_pri:
            for p in pre_by_pri[pri_us]:
                ax.plot(p['doppler_bin'],
                        p['range_bin'], 'yo', mfc='none', markersize=8, label='Pre-detect')
        
        # Plot CFAR detections (if any)
        if pri_us in det_by_pri:
            for d in det_by_pri[pri_us]:
                ax.plot(d['doppler_bin'],
                        d['range_bin'], 'ro', mfc='none', markersize=10, label='CFAR')
        
        # Set title based on whether it's a single map or multiple
        if pri_us == 'Single':
            ax.set_title("Range–Doppler Map")
        else:
            ax.set_title(f"{pri_us} µs")
            
        ax.set_xlabel("Doppler Bins")
        if i == 0:
            ax.set_ylabel("Range Bins")
    
    plt.tight_layout()
    plt.show()