# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:49:37 2026

@author: royi.b
Main script for radar signal processing and target detection.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tabulate import tabulate

# Import custom modules
from waveform import generate_pulse
from signal_processing import (
    build_tx_waveform,
    simulate_rx_waveform,
    build_data_cube,
    process_range_doppler
)
from detection import pre_detection, cfar_detection
from association import unfold_multiple_detections
from visualization import plot_rd_maps

plt.close('all')

# --- Constants ---
c = 3e8


def define_targets():
    """
    Define target parameters: (range_m, velocity_m/s, rcs)
    """
    return [
        (2000, 20, 1),
        (10000, 30, 1),
        (35000, -25, 1),
        (40000, 0, 1),
    ]


def main():
    """
    Main execution function for radar processing pipeline.
    """
    plt.close('all')
    # --- Waveform Parameters ---
    PRI_set = [39, 37, 34, 30.5, 27.5]  # µs
    Npulse = 2048  # number of pulses (slow time)
    Nrange = 1024  # range samples (fast time)
    waveform = 'barker'  # waveform type
    fc = 34.5e9
    fs = 60e6  # Tx sampling rate
    dec = 3  # decimation factor for Rx
    f_start = -fs/1e6  # LFM start frequency
    f_end = +fs/1e6  # LFM end frequency
    PWclks = 360  # PW in clocks
    
    # --- Detection and Association Parameters ---
    DR = 50  # dynamic range in dB
    max_zones = 12  # max zones for range unfolding
    tol_m = 100  # tolerance for range association (meter)
    TH1 = 13  # pre detection threshold   
    TH2 = 10  # CFAR threshold    
    
    # ---- Target Definition
    targets = define_targets()
    
    # ---- Pulse Generation
    pulse = generate_pulse(fs, waveform, PWclks, f_start, f_end)
    
    # ---- Processing Loop  
    plot_manager = [] # allocate plots list
    for i, PRI in enumerate(PRI_set, start=1):
        print(f"\nProcessing PRI {i} = {PRI} µs")
        
        # TX waveform
        tx_waveform, pulse_starts, pri_samples, PRIs, total_samples = \
            build_tx_waveform(pulse, [PRI], fs, Npulse)

        # RX signal with targets
        rx_waveform, pulse_windows = \
            simulate_rx_waveform(tx_waveform, pulse, pulse_starts, 
                                pri_samples, PRIs, fs, fc, targets)
        
        # Data cube
        data_cube, max_samples = \
            build_data_cube(rx_waveform, pulse_starts, pri_samples, Nrange, dec)

        # PRI grouping
        pri_groups = defaultdict(list)
        for j, pri_val in enumerate(PRIs):
            pri_groups[round(pri_val * 1e6, 1)].append(j)
            
        # Process RD map (CPI)
        RD_map = process_range_doppler(data_cube, PRIs, pri_groups, pulse, fs, c, fc, Nrange, dec)
        
        # ---- Detection
        pre_plots = pre_detection(RD_map, pri_groups, fs, c, fc, TH1, dwin=2, rwin=3)
        print("\n--- Detected Pre Plots ---")
        print(tabulate(pre_plots, headers="keys", tablefmt="pretty", floatfmt=".2f")) 
        plots = cfar_detection(RD_map, pre_plots, fs, c, fc, dec, Tr=3, Td=12, Gr=1, Gd=3, offset_dB=TH2)
        plot_manager.extend(plots)
        print("\n--- Detected Plots ---")
        print(tabulate(plot_manager, headers="keys", tablefmt="pretty", floatfmt=".2f"))   
        
        
        # ---- Association
        print('end debug')
        
        # ---- Visualization
        # plot_rd_maps(RD_map, DR, plot_manager, pre_plots)
    
    # ---- Association
    unfold_results = unfold_multiple_detections(plot_manager, max_zones, fc, tol_m=100)

    print("\n--- Associated Targets ---")
    for res in unfold_results:
        print(f"✅ Filtered range: {res['filt_rng']:.2f} m")
        print(f"✅ Filtered velocity: {res['filt_vel']:.2f} m/s")
        print(f"Used zone indices m: {res['zone_indices']}")
        print("PRI to folded range association:")
        for peak, filt_rng in zip(res['peaks'], res['filt_rng_list']):
            print(f"  PRI: {peak['PRI_us']} µs folded {peak['amb_rng']:.2f} km -> R_true: {filt_rng:.2f} m")
        print("\n")


if __name__ == "__main__":
    main()