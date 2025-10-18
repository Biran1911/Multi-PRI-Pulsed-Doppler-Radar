🧠 Overview:

    This Python script simulates a multi-PRI pulsed radar system using Barker-coded waveforms, generates synthetic echoes from defined targets, processes the data into Range–Doppler maps, performs detection            (CFAR), and then associates folded detections across different PRIs to estimate the true target range and velocity.

⚙️ Main Components:
1. Target and Waveform Definition

    define_targets() – defines target range (m), velocity (m/s), and RCS.

    generate_barker_pulse() – creates a complex baseband Barker code pulse at 1 MHz subcarrier frequency, sampled at fs.

2. Transmission and Reception Simulation

    build_tx_waveform() – builds the transmit signal for a given set of PRIs and number of pulses.

    simulate_rx_waveform() – generates a synthetic received signal by:

      Adding time-delayed, Doppler-shifted echoes for each target.

      Injecting additive complex Gaussian noise.

3. Data Cube Formation

    build_data_cube() – segments the received signal into pulses forming a 2D matrix (pulse × samples), preparing for FFT processing.

4. Range–Doppler Processing

    process_range_doppler() – performs:

      Range compression via matched filtering (FFT-based correlation).

      Doppler FFT to create a 2D Range–Doppler map for each PRI.

      Outputs magnitude in dB, velocity, and range axes per PRI.

5. Detection Stages

    pre_detection() – simple thresholding to select strong candidate cells in the Range–Doppler map.

    cfar_detection() – applies 2D CA-CFAR around candidate cells using training and guard windows.

      Converts to linear scale, estimates noise level, and sets detection threshold.

      Keeps detections where CUT > threshold.

6. Visualization

    plot_rd_maps() – displays Range–Doppler maps for all PRIs with overlaid pre-detections (yellow) and CFAR detections (red).

7. Association / True Range Resolution

    group_detections_by_doppler_freq() – groups detections with similar Doppler frequencies across PRIs.

    resolve_true_range() – tests combinations of zone indices (range-folds) across PRIs using Chinese Remainder-like reasoning to find a consistent true range.

    unfold_multiple_detections() – applies resolve_true_range() to all detection groups and reports final true range and velocity estimates.

🚀 Main Routine (main())

    Defines radar parameters (PRIs, sampling rate, carrier frequency, thresholds, etc.).

    Simulates and processes each PRI sequentially.

    Performs pre-detection and CFAR detection.

    Displays Range–Doppler plots.

    Resolves folded detections to estimate the true target range and velocity.

📊 Outputs

    Printed detection tables (before and after CFAR).

    Range–Doppler plots with detections.

    Final estimated true range and velocity of each target, including their PRI-to-range mapping consistency.
