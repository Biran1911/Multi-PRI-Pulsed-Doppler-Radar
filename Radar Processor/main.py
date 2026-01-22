# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:49:37 2026

@author: royi.b
Main script for radar signal processing and target detection.
"""
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
from association import ( 
    unfold_targets_from_plot_manager,
    update_target_manager
)
from visualization import plot_rd_maps

plt.close('all')

# --- Constants ---
c = 3e8


def define_targets():
    """
    Define target parameters: (range_m, velocity_m/s, rcs)
    """
    return [
        # (1000, 10, 1),
        # (2000, 20, 1),
        # (3000, 30, 1),
        (40000, 80, 1),
    ]


def main():
    """
    Main execution function for radar processing pipeline.
    """
    plt.close('all')
    # --- Waveform Parameters ---
    PRI_set = [39, 37, 34, 30.5, 27.5]  # µs
    # PRI_set = [40, 40, 40, 40, 40]  # µs
    Npulse = 2048  # number of pulses (slow time)
    waveform = 'barker'  # waveform type
    fc = 34.5e9
    fs = 60e6  # Tx sampling rate
    dec = 3  # decimation factor for Rx
    f_start = -fs/1e6  # LFM start frequency
    f_end = +fs/1e6  # LFM end frequency
    PWclks = 256  # PW in clocks
    
    # --- Detection and Association Parameters ---
    DR = 50  # dynamic range in dB
    TH1 = 15  # pre detection threshold   
    TH2 = 10  # CFAR threshold    
    
    # ---- Simulated Target Definition
    sim_targets = define_targets()
    
    # ---- Pulse Generation
    pulse = generate_pulse(fs, waveform, PWclks, f_start, f_end)
    
    # ---- Processing Loop  
    pre_plots = [] # allocate plots list
    plot_manager = [] # allocate plots list
    target_manager = [] # allocate targets list
    track_manager = [] # allocate track list
    cycle_idx = 0 # main loop cycle index (CPI time)
    
    for i, PRI in enumerate(PRI_set, start=1):
        print(f"\nProcessing PRI {i} = {PRI} µs")
        
        # TX waveform
        tx_waveform, pulse_starts, pri_samples, PRIs, total_samples = build_tx_waveform(pulse, [PRI], fs, Npulse)

        # RX signal with targets
        rx_waveform, pulse_windows = simulate_rx_waveform(tx_waveform, pulse, pulse_starts, pri_samples, PRIs, fs, fc, sim_targets)
        
        # Data cube
        data_cube, max_samples = build_data_cube(rx_waveform, pulse_windows)

        # PRI grouping
        pri_groups = defaultdict(list)
        for j, pri_val in enumerate(PRIs):
            pri_groups[round(pri_val * 1e6, 1)].append(j)
            
        # Process RD map (CPI)
        RD_map = process_range_doppler(data_cube, PRIs, pri_groups, pulse, fs, c, fc, dec)
        
        # ---- Detection
        pre_plots = pre_detection(RD_map, pri_groups, fs, c, fc, TH1, dwin=1, rwin=10)
        print("\n--- Detected Pre Plots ---")
        print(tabulate(pre_plots, headers="keys", tablefmt="pretty", floatfmt=".2f")) 
        plots = cfar_detection(RD_map, pre_plots, fs, c, fc, dec, Tr=80, Td=5, Gr=0, Gd=0, offset_dB=TH2)
        plot_manager.extend(plots)
        print("\n--- Detected Plots ---")
        print(tabulate(plot_manager, headers="keys", tablefmt="pretty", floatfmt=".2f"))   
        
        # ---- Visualization
        plot_rd_maps(RD_map, DR, plot_manager, pre_plots)
        
        # ---- Plot Association
        cycle_idx += 1 # new CPI index
        targets = unfold_targets_from_plot_manager(plots, fc=fc, r_tol=50.0, v_tol=2.0, min_pri_support=1)
        update_target_manager(target_manager, targets, cycle_idx)
        print("\n--- Associated Targets ---")
        for Trgt in target_manager:
            print(
                f"🎯 ID={Trgt['id']:2d} | "
                f"R={Trgt['filt_rng']:9.1f} m | "
                f"V={Trgt['filt_vel']:7.2f} m/s | "
                f"P={Trgt['power_dB']:6.1f} dB | "
                f"hits={Trgt['hits']:2d} | "
                f"age={Trgt['age']:2d}"
            )
            
    # ---- Track Association
    print("debug")
        
if __name__ == "__main__":
    main()