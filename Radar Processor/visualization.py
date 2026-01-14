# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:48:53 2026

@author: royi.b
Association module for target tracking and range unfolding.

Visualization module for plotting Range-Doppler maps and detections.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_rd_maps(RD_maps, DR, detections, pre_candidates, dec):
    """
    Plot Range-Doppler maps for each PRI and overlay CFAR detections.

    Parameters
    ----------
    RD_maps : dict
        Dictionary of {PRI_us: {'fft_db': ndarray, ...}}
    DR : float
        Dynamic range (dB) for color scale
    detections : list
        List of confirmed detections (from CFAR)
    pre_candidates : list
        List of pre-detection candidates (optional)
    dec : int
        Decimation factor
    """
    fig, axs = plt.subplots(1, len(RD_maps), figsize=(17, 8), sharey=True)

    # Group detections by PRI for fast lookup
    det_by_pri = {}
    if detections:
        for d in detections:
            det_by_pri.setdefault(d['PRI_us'], []).append(d)

    pre_by_pri = {}
    if pre_candidates:
        for p in pre_candidates:
            pre_by_pri.setdefault(p['PRI_us'], []).append(p)

    for i, (pri_us, data) in enumerate(sorted(RD_maps.items())):
        ax = axs[i]
        fft_db = data['fft_db']
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

        ax.set_title(f"{pri_us} µs")
        ax.set_xlabel("Doppler Bins")
        if i == 0:
            ax.set_ylabel("Range Bins")

    plt.tight_layout()
    plt.suptitle("Range–Doppler Maps with Detections", y=1.05, fontsize=16)
    plt.show()