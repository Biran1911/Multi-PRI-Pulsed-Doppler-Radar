**Multi-PRI Radar Signal Processing System for Range-Doppler Detection and Ambiguity Resolution
**
This project implements a comprehensive radar signal processing pipeline designed to detect and track multiple targets while resolving range and velocity ambiguities inherent in pulsed radar systems. The system employs a multi-PRI (Pulse Repetition Interval) staggered waveform strategy operating at 34.5 GHz with configurable pulse compression waveforms including Barker codes, Linear Frequency Modulation (LFM), and Maximum Length Sequences (MLS).
The processing chain encompasses several key stages: (1) baseband pulse generation with support for phase-coded and frequency-modulated waveforms; (2) radar echo simulation incorporating realistic target dynamics with range, velocity, and radar cross-section parameters; (3) signal decimation and matched filtering for pulse compression; (4) coherent integration via Doppler FFT to generate Range-Doppler maps for each PRI; (5) two-stage detection using threshold-based pre-detection followed by 2D Cell-Averaging CFAR (CA-CFAR) with configurable guard and training cells; and (6) multi-hypothesis association algorithm that resolves range ambiguities across multiple PRIs by testing zone combinations to determine true target range.
The system demonstrates robust performance in detecting four simulated targets across a 40 km range span with velocities ranging from -25 to +30 m/s. The multi-PRI approach (39, 37, 34, 30.5, and 27.5 µs) enables unambiguous range and velocity measurements by exploiting the Chinese Remainder Theorem principle, where different PRIs create different folding patterns that can be uniquely resolved. The modular Python implementation separates waveform generation, signal processing, detection, association, and visualization into distinct modules, facilitating maintenance, testing, and future enhancements.
Key technical specifications include: 60 MHz sampling rate with 3× decimation, 2048-pulse coherent processing intervals, configurable CFAR parameters (3 range training cells, 12 Doppler training cells, 10 dB threshold offset), and a maximum of 12 ambiguity zones for range unfolding with 100-meter tolerance. The system provides comprehensive visualization of Range-Doppler maps with detection overlays and detailed reporting of estimated target parameters including unfolded range and velocity.
Keywords: Pulse-Doppler radar, multi-PRI waveforms, CFAR detection, range ambiguity resolution, matched filtering, coherent processing, target association, radar signal processing

Application Domains: Air surveillance radar, automotive radar, missile guidance systems, weather radar, and any pulsed-Doppler radar system requiring extended unambiguous range coverage.
Technical Contributions:

Multi-waveform pulse compression implementation (Barker-13, MLS-63, LFM)
Integrated 2D CA-CFAR detector with candidate-based processing
Multi-PRI association algorithm with exhaustive zone search
Modular software architecture for radar signal processing research

Radar Processor Output:
Processing PRI 1 = 39 µs

Processing PRI 2 = 37 µs

Processing PRI 3 = 34 µs

Processing PRI 4 = 30.5 µs

Processing PRI 5 = 27.5 µs

--- Detected Pre Plots ---
+--------+-------------+-----------+----------+
| PRI_us | doppler_bin | range_bin | power_dB |
+--------+-------------+-----------+----------+
|  39.0  |    -459     |    767    |  80.01   |
|  39.0  |      0      |    654    |  82.31   |
|  39.0  |     367     |    267    |  79.76   |
|  39.0  |     551     |    554    |  82.13   |
|  37.0  |    -436     |    227    |  80.78   |
|  37.0  |      0      |    154    |  81.99   |
|  37.0  |     349     |    267    |  79.19   |
|  37.0  |     523     |    594    |   81.7   |
|  34.0  |    -400     |    587    |  78.09   |
|  34.0  |      0      |    574    |  80.29   |
|  34.0  |     320     |    267    |  78.93   |
|  34.0  |     480     |    654    |  74.79   |
|  30.5  |    -359     |    397    |  81.06   |
|  30.5  |      0      |    454    |  81.47   |
|  30.5  |     287     |    267    |  79.84   |
|  30.5  |     431     |    114    |  81.49   |
|  27.5  |    -324     |    267    |  80.81   |
|  27.5  |      0      |    384    |  81.17   |
|  27.5  |     259     |    267    |  81.13   |
|  27.5  |     389     |    234    |  78.88   |
+--------+-------------+-----------+----------+

--- Detected Plots ---
+--------+-------------+-----------+----------+--------------+----------+-----------------+-------------------+--------+
| PRI_us | doppler_bin | range_bin | power_dB | doppler_freq | velocity | folded_range_km | CFAR_threshold_dB | CUT_dB |
+--------+-------------+-----------+----------+--------------+----------+-----------------+-------------------+--------+
|  39.0  |    -459     |    767    |  80.01   |   -5746.69   |  -24.99  |     5.7525      |       55.83       | 80.01  |
|  39.0  |      0      |    654    |  82.31   |     0.0      |   0.0    |      4.905      |       47.87       | 82.31  |
|  39.0  |     367     |    267    |  79.76   |   4594.85    |  19.98   |     2.0025      |       55.28       | 79.76  |
|  39.0  |     551     |    554    |  82.13   |   6898.54    |  29.99   |      4.155      |       49.96       | 82.13  |
|  37.0  |    -436     |    227    |  80.78   |   -5753.8    |  -25.02  |     1.7025      |       53.57       | 80.78  |
|  37.0  |      0      |    154    |  81.99   |     0.0      |   0.0    |      1.155      |       47.59       | 81.99  |
|  37.0  |     349     |    267    |  79.19   |   4605.68    |  20.02   |     2.0025      |       55.08       | 79.19  |
|  37.0  |     523     |    594    |   81.7   |   6901.92    |  30.01   |      4.455      |       50.43       |  81.7  |
|  34.0  |    -400     |    587    |  78.09   |   -5744.49   |  -24.98  |     4.4025      |       53.06       | 78.09  |
|  34.0  |      0      |    574    |  80.29   |     0.0      |   0.0    |      4.305      |       45.92       | 80.29  |
|  34.0  |     320     |    267    |  78.93   |   4595.59    |  19.98   |     2.0025      |       52.17       | 78.93  |
|  34.0  |     480     |    654    |  74.79   |   6893.38    |  29.97   |      4.905      |       56.18       | 74.79  |
|  30.5  |    -359     |    397    |  81.06   |   -5747.31   |  -24.99  |     2.9775      |       50.53       | 81.06  |
|  30.5  |      0      |    454    |  81.47   |     0.0      |   0.0    |      3.405      |       47.1        | 81.47  |
|  30.5  |     287     |    267    |  79.84   |   4594.65    |  19.98   |     2.0025      |       53.71       | 79.84  |
|  30.5  |     431     |    114    |  81.49   |   6899.97    |   30.0   |      0.855      |       47.04       | 81.49  |
|  27.5  |    -324     |    267    |  80.81   |   -5752.84   |  -25.01  |     2.0025      |       50.04       | 80.81  |
|  27.5  |      0      |    384    |  81.17   |     0.0      |   0.0    |      2.88       |       46.85       | 81.17  |
|  27.5  |     259     |    267    |  81.13   |   4598.72    |  19.99   |     2.0025      |       47.67       | 81.13  |
|  27.5  |     389     |    234    |  78.88   |   6906.96    |  30.03   |      1.755      |       54.02       | 78.88  |
+--------+-------------+-----------+----------+--------------+----------+-----------------+-------------------+--------+

--- Associated Targets ---
✅ Estimated range: 35.00 km
✅ Estimated velocity: -25.00 m/s
Used zone indices m: (5, 6, 6, 7, 8)
PRI to folded range association:
  PRI: 39.0 µs folded 5.75 km -> R_true: 35.002 km
  PRI: 37.0 µs folded 1.70 km -> R_true: 35.002 km
  PRI: 34.0 µs folded 4.40 km -> R_true: 35.002 km
  PRI: 30.5 µs folded 2.98 km -> R_true: 35.002 km
  PRI: 27.5 µs folded 2.00 km -> R_true: 35.002 km


✅ Estimated range: 40.01 km
✅ Estimated velocity: 0.00 m/s
Used zone indices m: (6, 7, 7, 8, 9)
PRI to folded range association:
  PRI: 39.0 µs folded 4.91 km -> R_true: 40.005 km
  PRI: 37.0 µs folded 1.16 km -> R_true: 40.005 km
  PRI: 34.0 µs folded 4.30 km -> R_true: 40.005 km
  PRI: 30.5 µs folded 3.40 km -> R_true: 40.005 km
  PRI: 27.5 µs folded 2.88 km -> R_true: 40.005 km


✅ Estimated range: 2.00 km
✅ Estimated velocity: 20.00 m/s
Used zone indices m: (0, 0, 0, 0, 0)
PRI to folded range association:
  PRI: 39.0 µs folded 2.00 km -> R_true: 2.002 km
  PRI: 37.0 µs folded 2.00 km -> R_true: 2.002 km
  PRI: 34.0 µs folded 2.00 km -> R_true: 2.002 km
  PRI: 30.5 µs folded 2.00 km -> R_true: 2.002 km
  PRI: 27.5 µs folded 2.00 km -> R_true: 2.002 km


✅ Estimated range: 10.01 km
✅ Estimated velocity: 30.00 m/s
Used zone indices m: (1, 1, 1, 2, 2)
PRI to folded range association:
  PRI: 39.0 µs folded 4.16 km -> R_true: 10.005 km
  PRI: 37.0 µs folded 4.46 km -> R_true: 10.005 km
  PRI: 34.0 µs folded 4.91 km -> R_true: 10.005 km
  PRI: 30.5 µs folded 0.85 km -> R_true: 10.005 km
  PRI: 27.5 µs folded 1.75 km -> R_true: 10.005 km
