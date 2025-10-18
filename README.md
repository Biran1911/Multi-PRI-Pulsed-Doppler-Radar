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

    Processing PRI 1 = 44.5 µs

    Processing PRI 2 = 40 µs
    
    Processing PRI 3 = 38 µs
    
    Processing PRI 4 = 34.5 µs
    
    Processing PRI 5 = 31 µs
    
    --- Detected Pre Plots ---
    +--------+-------------+-----------+--------------------+
    | PRI_us | doppler_bin | range_bin |      power_dB      |
    +--------+-------------+-----------+--------------------+
    |  44.5  |   -524.0    |    209    | 33.25339350203012  |
    |  44.5  |   -524.0    |    212    | 30.751068726350645 |
    |  44.5  |   -524.0    |    213    | 33.116651682376386 |
    |  44.5  |   -524.0    |    216    | 43.42382501073354  |
    |  44.5  |   -524.0    |    219    | 33.45837555154303  |
    |  44.5  |   -524.0    |    220    |  30.9361887710762  |
    |  44.5  |   -524.0    |    223    | 33.30716840746711  |
    |  44.5  |     0.0     |    259    | 33.33447225454777  |
    |  44.5  |     0.0     |    262    | 30.93607467327751  |
    |  44.5  |     0.0     |    263    | 32.94388975607295  |
    |  44.5  |     0.0     |    266    | 43.55428131387192  |
    |  44.5  |     0.0     |    269    | 33.16925712646804  |
    |  44.5  |     0.0     |    270    | 30.919763530625033 |
    |  44.5  |     0.0     |    273    | 33.45954096283699  |
    |  44.5  |    629.0    |    636    | 32.612208724911675 |
    |  44.5  |    629.0    |    640    |  33.0317925303117  |
    |  44.5  |    629.0    |    643    | 43.15001215072194  |
    |  44.5  |    629.0    |    646    | 32.94342789249642  |
    |  44.5  |    629.0    |    650    | 33.05574735991227  |
    |  40.0  |   -471.0    |    671    | 39.531406319875146 |
    |  40.0  |   -470.0    |    671    | 41.26909118549209  |
    |  40.0  |   -469.0    |    671    | 31.341770949645237 |
    |  40.0  |     0.0     |    259    | 33.95664676266894  |
    |  40.0  |     0.0     |    262    | 31.79491841011586  |
    |  40.0  |     0.0     |    263    | 33.78296471270241  |
    |  40.0  |     0.0     |    266    |  44.3165294481011  |
    |  40.0  |     0.0     |    269    | 34.08060717412664  |
    |  40.0  |     0.0     |    270    | 31.67799369476421  |
    |  40.0  |     0.0     |    273    | 34.32347234769849  |
    |  40.0  |    564.0    |    734    | 39.49184529588458  |
    |  40.0  |    565.0    |    734    |  41.1265423822031  |
    |  38.0  |   -447.0    |    105    | 34.74004905658085  |
    |  38.0  |   -447.0    |    108    | 32.122955302186455 |
    |  38.0  |   -447.0    |    109    | 34.323872457929774 |
    |  38.0  |   -447.0    |    112    | 44.70576470390615  |
    |  38.0  |   -447.0    |    115    | 34.759290879649434 |
    |  38.0  |   -447.0    |    116    | 31.926222032630992 |
    |  38.0  |   -447.0    |    119    | 34.88663161849932  |
    |  38.0  |     0.0     |    259    | 34.652782708525656 |
    |  38.0  |     0.0     |    262    |  32.1334595984144  |
    |  38.0  |     0.0     |    263    | 34.58179190054302  |
    |  38.0  |     0.0     |    266    | 44.88025319564415  |
    |  38.0  |     0.0     |    269    | 34.71873236491446  |
    |  38.0  |     0.0     |    270    | 32.158225688131985 |
    |  38.0  |     0.0     |    273    | 34.888369501329166 |
    |  38.0  |    536.0    |     8     | 33.533958093312336 |
    |  38.0  |    536.0    |    12     | 33.40204509611667  |
    |  38.0  |    536.0    |    15     | 43.64171275228357  |
    |  38.0  |    536.0    |    18     |  33.2163006735715  |
    |  38.0  |    536.0    |    22     | 33.326989094380636 |
    |  38.0  |    537.0    |    15     | 35.45748326989075  |
    |  34.5  |   -407.0    |    526    | 36.10872411938506  |
    |  34.5  |   -406.0    |    519    | 34.44937733050076  |
    |  34.5  |   -406.0    |    523    | 34.28476342874937  |
    |  34.5  |   -406.0    |    526    | 44.52580203447936  |
    |  34.5  |   -406.0    |    529    | 34.239209969179136 |
    |  34.5  |   -406.0    |    533    | 34.07457148509281  |
    |  34.5  |     0.0     |    259    | 35.372675089767725 |
    |  34.5  |     0.0     |    263    | 35.47358915378175  |
    |  34.5  |     0.0     |    266    | 45.652295221612896 |
    |  34.5  |     0.0     |    269    | 35.44919864851487  |
    |  34.5  |     0.0     |    270    | 32.83299752848053  |
    |  34.5  |     0.0     |    273    | 35.36723712861622  |
    |  34.5  |    487.0    |    153    | 41.18525500360036  |
    |  34.5  |    488.0    |    153    | 42.16107325940065  |
    |  31.0  |   -365.0    |    319    | 36.39509619940604  |
    |  31.0  |   -365.0    |    322    | 34.097336532406416 |
    |  31.0  |   -365.0    |    323    |  36.1366404553491  |
    |  31.0  |   -365.0    |    326    | 46.571314834898374 |
    |  31.0  |   -365.0    |    329    | 36.34614934614437  |
    |  31.0  |   -365.0    |    330    | 33.69356938152225  |
    |  31.0  |   -365.0    |    333    |  36.0437164087378  |
    |  31.0  |     0.0     |    259    | 36.583630975035874 |
    |  31.0  |     0.0     |    262    | 33.988535856471515 |
    |  31.0  |     0.0     |    263    | 36.459412077866496 |
    |  31.0  |     0.0     |    266    | 46.65186923331167  |
    |  31.0  |     0.0     |    269    | 36.33526931004241  |
    |  31.0  |     0.0     |    273    | 36.429574319584034 |
    |  31.0  |    438.0    |    286    | 36.43483211631895  |
    |  31.0  |    438.0    |    289    | 34.019234435269205 |
    |  31.0  |    438.0    |    290    | 36.19177158649522  |
    |  31.0  |    438.0    |    293    | 46.565873325866384 |
    |  31.0  |    438.0    |    296    | 36.32303845865335  |
    |  31.0  |    438.0    |    297    | 33.75935660419667  |
    |  31.0  |    438.0    |    300    | 36.21126209179734  |
    +--------+-------------+-----------+--------------------+
    
    --- Detected Plots ---
    +--------+-------------+-----------+--------------------+--------------------+---------------------+-----------------+--------------------+--------------------+
    | PRI_us | doppler_bin | range_bin |      power_dB      |    doppler_freq    |      velocity       | folded_range_km | CFAR_threshold_dB  |       CUT_dB       |
    +--------+-------------+-----------+--------------------+--------------------+---------------------+-----------------+--------------------+--------------------+
    |  44.5  |   -524.0    |    216    | 43.42382501073354  | -5749.648876404494 | -24.998473375671715 |      1.62       | 39.198552456035145 | 43.42382501073354  |
    |  44.5  |     0.0     |    266    | 43.55428131387192  |        0.0         |         0.0         |      1.995      | 39.181818991277694 | 43.55428131387192  |
    |  44.5  |    629.0    |    643    | 43.15001215072194  | 6901.773174157303  |  30.00770945285784  |     4.8225      | 39.31028195702616  | 43.15001215072194  |
    |  40.0  |   -470.0    |    671    | 41.26909118549209  | -5737.304687500001 | -24.944802989130437 |     5.0325      | 40.79089522139317  | 41.26909118549209  |
    |  40.0  |     0.0     |    266    |  44.3165294481011  |        0.0         |         0.0         |      1.995      | 39.95531418804339  |  44.3165294481011  |
    |  40.0  |    565.0    |    734    |  41.1265423822031  | 6896.972656250001  |  29.98683763586957  |      5.505      |  40.6181102757618  |  41.1265423822031  |
    |  38.0  |   -447.0    |    112    | 44.70576470390615  | -5743.729440789474 | -24.97273669908467  |      0.84       | 40.74987158823647  | 44.70576470390615  |
    |  38.0  |     0.0     |    266    | 44.88025319564415  |        0.0         |         0.0         |      1.995      | 40.602258367140024 | 44.88025319564415  |
    |  34.5  |   -406.0    |    526    | 44.52580203447936  | -5746.150362318841 | -24.983262444864526 |      3.945      | 41.713381324004146 | 44.52580203447936  |
    |  34.5  |     0.0     |    266    | 45.652295221612896 |        0.0         |         0.0         |      1.995      |  41.2598437979738  | 45.652295221612896 |
    |  34.5  |    488.0    |    153    | 42.16107325940065  | 6906.702898550725  | 30.029143037177064  |     1.1475      | 41.88180192419297  | 42.16107325940065  |
    |  31.0  |   -365.0    |    326    | 46.571314834898374 | -5749.117943548386 | -24.996164971949504 |      2.445      | 42.180152132296165 | 46.571314834898374 |
    |  31.0  |     0.0     |    266    | 46.65186923331167  |        0.0         |         0.0         |      1.995      | 42.379084332111475 | 46.65186923331167  |
    |  31.0  |    438.0    |    293    | 46.565873325866384 | 6898.941532258064  |  29.99539796633941  |     2.1975      | 42.137274727608315 | 46.565873325866384 |
    +--------+-------------+-----------+--------------------+--------------------+---------------------+-----------------+--------------------+--------------------+
    
    --- Associated Targets ---
    ✅ Estimated range: 35.01 km
    ✅ Estiamted velocity: -25.00 m/s
    Used zone indices m: (5, 5, 6, 6, 7)
    PRI to folded range association:
      PRI: 44.5 µs folded 1.62 km -> R_true: 34.995 km
      PRI: 40.0 µs folded 5.03 km -> R_true: 35.032 km
      PRI: 38.0 µs folded 0.84 km -> R_true: 35.040 km
      PRI: 34.5 µs folded 3.94 km -> R_true: 34.995 km
      PRI: 31.0 µs folded 2.44 km -> R_true: 34.995 km
    
    
    ✅ Estimated range: 2.00 km
    ✅ Estiamted velocity: 0.00 m/s
    Used zone indices m: (0, 0, 0, 0, 0)
    PRI to folded range association:
      PRI: 44.5 µs folded 2.00 km -> R_true: 1.995 km
      PRI: 40.0 µs folded 2.00 km -> R_true: 1.995 km
      PRI: 38.0 µs folded 2.00 km -> R_true: 1.995 km
      PRI: 34.5 µs folded 2.00 km -> R_true: 1.995 km
      PRI: 31.0 µs folded 2.00 km -> R_true: 1.995 km
    
    
    ✅ Estimated range: 11.50 km
    ✅ Estiamted velocity: 30.00 m/s
    Used zone indices m: (1, 1, 2, 2)
    PRI to folded range association:
      PRI: 44.5 µs folded 4.82 km -> R_true: 11.497 km
      PRI: 40.0 µs folded 5.50 km -> R_true: 11.505 km
      PRI: 34.5 µs folded 1.15 km -> R_true: 11.497 km
      PRI: 31.0 µs folded 2.20 km -> R_true: 11.498 km
<img width="1920" height="983" alt="image" src="https://github.com/user-attachments/assets/f3049230-aa06-468c-983a-f7734fa2c40e" />

