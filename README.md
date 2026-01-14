# Multi-PRI Radar Signal Processing System

A comprehensive Python-based radar signal processing pipeline for detecting and tracking multiple targets while resolving range and velocity ambiguities in pulsed radar systems.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## Overview

This project implements a multi-PRI (Pulse Repetition Interval) radar system operating at 34.5 GHz with advanced signal processing capabilities. The system detects and tracks targets while resolving range and velocity ambiguities through sophisticated association algorithms.

## Key Features

- **Multiple Waveform Support**: Barker-13, MLS-63, and Linear Frequency Modulation (LFM)
- **Multi-PRI Processing**: Five staggered PRIs (39, 37, 34, 30.5, 27.5 µs) for ambiguity resolution
- **Advanced Detection**: Two-stage detection with pre-detection and 2D CA-CFAR
- **Range Unfolding**: Multi-hypothesis association algorithm for true range estimation
- **Modular Architecture**: Clean separation of concerns across six modules
- **Visualization**: Comprehensive Range-Doppler map plotting with detection overlays

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| **Operating Frequency** | 34.5 GHz |
| **Sampling Rate** | 60 MHz (TX) / 20 MHz (RX with decimation) |
| **PRIs** | 39, 37, 34, 30.5, 27.5 µs |
| **Coherent Integration** | 2048 pulses |
| **Pulse Width** | 360 clock cycles (~6 µs) |
| **Max Unambiguous Range** | Variable per PRI (5.85 - 11.25 km folded) |
| **CFAR Configuration** | 3×12 training cells, 1×3 guard cells |

## Project Structure

```
radar_project/
│
├── main.py                 # Main execution script
├── waveform.py            # Pulse generation (Barker, LFM, MLS)
├── signal_processing.py   # TX/RX simulation and Range-Doppler processing
├── detection.py           # Pre-detection and CFAR algorithms
├── association.py         # Target association and range unfolding
├── visualization.py       # Range-Doppler map plotting
└── README.md             # This file
```

## Processing Pipeline

```mermaid
graph LR
    A[Pulse Generation] --> B[TX Waveform Build]
    B --> C[RX Simulation]
    C --> D[Data Cube]
    D --> E[Range-Doppler Processing]
    E --> F[Pre-Detection]
    F --> G[CFAR Detection]
    G --> H[Multi-PRI Association]
    H --> I[Target Reports]
```

## Installation

### Requirements

```bash
pip install numpy matplotlib tabulate
```

### Python Version
- Python 3.10 or higher recommended

## Usage

### Basic Example

```python
from main import main

# Run the complete radar processing pipeline
if __name__ == "__main__":
    main()
```

### Customizing Parameters

Edit parameters in `main.py`:

```python
# Waveform Parameters
PRI_set = [39, 37, 34, 30.5, 27.5]  # µs
waveform = 'mls'  # Options: 'barker', 'lfm', 'mls'
fc = 34.5e9       # Carrier frequency
fs = 60e6         # Sampling rate

# Detection Parameters
TH1 = 13  # Pre-detection threshold (dB below peak)
TH2 = 10  # CFAR threshold offset (dB)

# Target Definition
targets = [
    (range_m, velocity_m/s, rcs),
    # Add more targets...
]
```

## Algorithm Details

### 1. Pulse Compression
- Matched filtering with decimated sampling (3×)
- Support for phase-coded (Barker, MLS) and chirp (LFM) waveforms
- Range bin alignment compensation

### 2. Coherent Integration
- Doppler FFT across pulse train
- Per-PRI processing for multi-PRI systems
- Logarithmic scaling for visualization

### 3. CFAR Detection
- 2D Cell-Averaging CFAR (CA-CFAR)
- Configurable guard and training cells
- Candidate-based processing for efficiency

### 4. Range Unfolding
- Groups detections by Doppler frequency (±50 Hz tolerance)
- Exhaustive search across ambiguity zones (up to 12 zones)
- Chinese Remainder Theorem-based resolution
- Consistency check with configurable tolerance (100 m)

## Performance

### Demonstrated Capabilities

| Metric | Value |
|--------|-------|
| **Targets Detected** | 4/4 (100%) |
| **Range Coverage** | 2 - 40 km |
| **Velocity Range** | -25 to +30 m/s |
| **Range Accuracy** | ±100 m (limited by range bin resolution) |
| **Processing Time** | ~2-3 seconds (5 PRIs × 2048 pulses) |

### Example Output

```
✅ Estimated range: 35.02 km
✅ Estimated velocity: -25.15 m/s
Used zone indices m: (3, 4, 5, 6, 7)
PRI to folded range association:
  PRI: 39.0 µs folded 5.85 km -> R_true: 35.019 km
  PRI: 37.0 µs folded 5.55 km -> R_true: 35.020 km
  ...
```

## Applications

- **Air Surveillance Radar**: Long-range aircraft tracking
- **Automotive Radar**: Advanced driver assistance systems (ADAS)
- **Missile Guidance**: Target tracking and fire control
- **Weather Radar**: Precipitation velocity estimation
- **Research & Education**: Radar signal processing demonstrations

## Contributing

Contributions are welcome! Areas for enhancement:

- Additional waveform types (OFDM, Costas codes)
- Track-before-detect algorithms
- Real-time processing optimization
- Hardware-in-the-loop integration
- Extended target modeling

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Royi Biran - 

## Acknowledgments

- Based on modern pulse-Doppler radar principles
- Implements standard CFAR detection algorithms
- Multi-PRI technique inspired by Chinese Remainder Theorem

## References

1. Richards, M. A. (2014). *Fundamentals of Radar Signal Processing*. McGraw-Hill Education.
2. Skolnik, M. I. (2008). *Radar Handbook*. McGraw-Hill Education.
3. Mahafza, B. R. (2013). *Radar Systems Analysis and Design Using MATLAB*. CRC Press.

---

**Keywords:** Pulse-Doppler radar, multi-PRI, CFAR detection, range ambiguity resolution, matched filtering, coherent processing, Python radar
