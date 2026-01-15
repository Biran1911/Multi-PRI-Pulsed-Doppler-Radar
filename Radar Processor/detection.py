# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:48:12 2026

@author: royi.b
Detection module for CFAR and pre-detection processing.
"""
import numpy as np


def pre_detection(RD_maps, pri_groups, fs, c, fc, TH1, dwin=1, rwin=1):
    """
    Detect strong peaks in each RD map and check local maximum.
    
    Parameters
    ----------
    RD_maps : dict
        Dictionary of range-Doppler maps
    pri_groups : dict
        PRI grouping information
    fs : float
        Sampling frequency [Hz]
    c : float
        Speed of light [m/s]
    fc : float
        Carrier frequency [Hz]
    TH1 : float
        Pre-detection threshold in dB below peak
    dwin : int
        Doppler window size for local maximum check
    rwin : int
        Range window size for local maximum check
        
    Returns
    -------
    candidates : list
        List of candidate detections
    """
    candidates = []

    for pri_us, data in RD_maps.items():
        fft_db = data['fft_db']
        peak_threshold = np.max(fft_db) - TH1

        Nd, Nr = fft_db.shape

        for d_bin in range(Nd):
            for r_bin in range(Nr):
                power = fft_db[d_bin, r_bin]

                if power < peak_threshold:
                    continue

                # neighborhood bounds
                d0 = max(0, d_bin - dwin)
                d1 = min(Nd, d_bin + dwin + 1)
                r0 = max(0, r_bin - rwin)
                r1 = min(Nr, r_bin + rwin + 1)

                neighborhood = fft_db[d0:d1, r0:r1]

                # local maximum test
                if power < np.max(neighborhood):
                    continue

                candidates.append({
                    'PRI_us': pri_us,
                    'doppler_bin': int(d_bin - Nd // 2),
                    'range_bin': int(r_bin),
                    'power_dB': np.round(power, 2),
                })

    return candidates


def cfar_detection(RD_maps, candidates, fs, c, fc, dec, Tr, Td, Gr, Gd, offset_dB):
    """
    Candidate-based 2D Cell Averaging CFAR (CA-CFAR).

    Parameters
    ----------
    RD_maps : dict
        Dictionary of {PRI_us: {'fft_db': 2D array, ...}}
    candidates : list of dict
        Each dict must contain keys: 'PRI_us', 'range_bin', 'doppler_bin'
    fs, c, fc : float
        Radar parameters
    dec : int
        Decimation factor
    Tr, Td : int
        Training cells (range, doppler)
    Gr, Gd : int
        Guard cells (range, doppler)
    offset_dB : float
        Threshold offset in dB

    Returns
    -------
    detections : list of dict
        Candidates that pass CFAR threshold
    """
    detections = []

    for cand in candidates:
        pri_us = cand['PRI_us']
        if pri_us not in RD_maps:
            continue

        RDM_dB = RD_maps[pri_us]['fft_db']
        Nd, Nr = RDM_dB.shape

        # Candidate indices
        i_range = int(cand['range_bin'])
        j_doppler = int((cand['doppler_bin'] + Nd//2) % Nd)

        # Skip edges
        if (i_range < Tr + Gr or i_range >= Nr - (Tr + Gr) or
            j_doppler < Td + Gd or j_doppler >= Nd - (Td + Gd)):
            continue

        # Linear scale
        RDM_linear = 10 ** (RDM_dB / 10)

        # Extract window around CUT
        window = RDM_linear[j_doppler - (Td + Gd): j_doppler + (Td + Gd) + 1,
                            i_range - (Tr + Gr): i_range + (Tr + Gr) + 1]

        # Guard mask
        guard_mask = np.zeros_like(window, dtype=bool)
        guard_mask[Td:Td + 2*Gd + 1, Tr:Tr + 2*Gr + 1] = True

        # Training cells only
        training_cells = window[~guard_mask]
        if len(training_cells) == 0:
            continue

        noise_level = np.mean(training_cells)
        threshold_dB = 10 * np.log10(noise_level) + offset_dB
        CUT_dB = RDM_dB[j_doppler, i_range]

        # CFAR decision
        if CUT_dB > threshold_dB:
            doppler_freq = np.fft.fftshift(np.fft.fftfreq(Nd, d=pri_us * 1e-6))[j_doppler]
            velocity = doppler_freq * c / (2 * fc)
            folded_range = (dec * (i_range / fs) * c / 2)  # m
          
            cand_out = cand.copy()
            cand_out['doppler_freq'] = np.round(doppler_freq, 2) 
            cand_out['amb_vel'] = np.round(velocity, 2)
            cand_out['amb_rng'] = np.round(folded_range, 2)
            cand_out['CFAR_threshold_dB'] = np.round(threshold_dB, 2)
            cand_out['CUT_dB'] = np.round(CUT_dB, 2)
            detections.append(cand_out)

    return detections