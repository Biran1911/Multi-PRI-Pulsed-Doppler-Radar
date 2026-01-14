# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:48:34 2026

@author: royi.b
Association module for target tracking and range unfolding.
"""
import numpy as np
from collections import defaultdict
from itertools import product


def group_detections_by_doppler_freq(detections, freq_tol_Hz=50):
    """
    Group detected peaks by Doppler frequency (not bin index),
    merging close frequencies within freq_tol_Hz.
    
    Parameters
    ----------
    detections : list
        List of detection dictionaries
    freq_tol_Hz : float
        Frequency tolerance in Hz for grouping
        
    Returns
    -------
    reduced_groups : dict
        Dictionary mapping frequency keys to detection lists
    """
    groups = defaultdict(list)

    for detection in detections:
        freq = detection['doppler_freq']
        freq_key = round(freq / freq_tol_Hz) * freq_tol_Hz
        groups[freq_key].append(detection)

    # Reduce to one peak per PRI per Doppler frequency (max power)
    reduced_groups = defaultdict(list)
    for freq_key, peaks in groups.items():
        by_pri = defaultdict(list)
        for p in peaks:
            by_pri[p['PRI_us']].append(p)
        # keep strongest per PRI
        for pri, plist in by_pri.items():
            best = max(plist, key=lambda x: x['power_dB'])
            reduced_groups[freq_key].append(best)
            
    return reduced_groups


def resolve_true_range(pris_us, folded_ranges_km, max_zones, tol_km):
    """
    Resolve true range for a target folded across multiple PRIs,
    allowing different zone index (m) per PRI.

    Parameters
    ----------
    pris_us : list
        List of PRIs in microseconds
    folded_ranges_km : list
        List of measured folded ranges (in km)
    max_zones : int
        Maximum number of zones to check for each PRI
    tol_km : float
        Tolerance in km for acceptable spread in computed R_true

    Returns
    -------
    dict with fields:
        'R_true'       : Estimated true range (km)
        'zone_indices' : List of m_i used for each PRI
        'r_true_list'  : Computed R_true per PRI
        'success'      : True if consistent R_true found, else False
    """
    c = 3e8
    N = len(pris_us)

    # Generate all possible m combinations
    for m_combo in product(range(max_zones), repeat=N):
        r_true_list = []
        for i in range(N):
            pri_sec = pris_us[i] * 1e-6
            R_fold_km = folded_ranges_km[i]
            R_zone_km = m_combo[i] * (c * pri_sec / 2) * 1e-3
            r_true = R_zone_km + R_fold_km
            r_true_list.append(r_true)

        if np.ptp(r_true_list) < tol_km:
            return {
                'R_true': np.mean(r_true_list),
                'zone_indices': m_combo,
                'r_true_list': r_true_list,
                'success': True
            }

    return {
        'R_true': None,
        'zone_indices': None,
        'r_true_list': [],
        'success': False
    }


def unfold_multiple_detections(detections, max_zones, fc, tol_km):
    """
    Unfold range ambiguities for multiple detections across PRIs.
    
    Parameters
    ----------
    detections : list
        List of detection dictionaries
    max_zones : int
        Maximum number of ambiguity zones to check
    fc : float
        Carrier frequency [Hz]
    tol_km : float
        Range tolerance in km
        
    Returns
    -------
    results : list
        List of unfolded target results
    """
    c = 3e8
    groups = group_detections_by_doppler_freq(detections)
    results = []

    for doppler_freq, peaks in groups.items():
        pris_us = [p['PRI_us'] for p in peaks]
        folded_ranges_km = [p['folded_range_km'] for p in peaks]

        result = resolve_true_range(pris_us, folded_ranges_km, max_zones, tol_km)
        if result['success']:
            results.append({
                'Velocity': (doppler_freq * c) / (2 * fc),
                'R_true': result['R_true'],
                'zone_indices': result['zone_indices'],
                'r_true_list': result['r_true_list'],
                'peaks': peaks
            })

    return results