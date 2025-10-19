import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tabulate import tabulate
from itertools import product

plt.close('all')

# --- Constants ---
c = 3e8

# --- Target definition ---
def define_targets():
    return [
        (2000, 0, 100),
        (11500, 30, 100),
        (35000, -25, 100),
        (40000, 10, 100),
    ]

# --- Generate baseband pulse ---
def generate_pulse(fs, waveform, nbin, f_start, f_end):
    """
    Generate Barker or LFM pulse waveform.

    Parameters
    ----------
    fs : float
        Sampling frequency [Hz]
    waveform : str
        Waveform type: 'barker_13', 'lfm', etc.
    nbin : float
        Range bin or pulse length parameter (depends on design)
    f_start : float, optional
        Start frequency for LFM [Hz]
    f_end : float, optional
        End frequency for LFM [Hz]
    pulse_duration : float, optional
        Pulse duration in seconds for LFM. If None, derived from barker_bin/fs.
    """
    # --- Handle Barker coded waveforms ---
    match waveform.lower():
        case 'barker_2':
            code = np.array([+1, -1])
        case 'barker_3':
            code = np.array([+1, +1, -1])
        case 'barker_4':
            code = np.array([+1, +1, -1, +1])
        case 'barker_5':
            code = np.array([+1, +1, +1, -1, +1])
        case 'barker_7':
            code = np.array([+1, +1, +1, -1, -1, +1, -1])
        case 'barker_11':
            code = np.array([+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1])
        case 'barker_13':
            code = np.array([+1, +1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, +1])
        case 'lfm':
            # --- Generate LFM waveform ---
            pulse_duration = nbin / fs  # fallback based on same logic
            t = np.arange(0, pulse_duration, 1/fs)
            k = (f_end - f_start) / pulse_duration  # chirp rate
            phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
            pulse_complex = np.exp(1j * phase)
            return pulse_complex
        case _:
            raise ValueError(f"Unsupported waveform: {waveform}")

    # --- Generate Barker pulse samples ---
    chip_duration = (nbin / fs) / len(code)
    chip_samples = int(fs * chip_duration)
    pulse_real = np.repeat(code, chip_samples)
    t_pulse = np.arange(len(pulse_real)) / fs
    pulse_complex = pulse_real * np.exp(1j * 2 * np.pi * 1e6 * t_pulse)
    return pulse_complex

# --- Build transmit waveform and PRI schedule ---
def build_tx_waveform(pulse, PRI_set, fs, Npulse):
    PRIs = np.tile(np.array(PRI_set) * 1e-6, Npulse)
    pri_samples = [int(pri * fs) for pri in PRIs]
    total_samples = sum(pri_samples)
    tx_waveform = np.zeros(total_samples, dtype=complex)
    pulse_starts = []

    cursor = 0
    for pri_len in pri_samples:
        tx_waveform[cursor:cursor + len(pulse)] = pulse
        pulse_starts.append(cursor)
        cursor += pri_len

    return tx_waveform, pulse_starts, pri_samples, PRIs, total_samples

# --- Simulate received waveform with injected echoes ---
def simulate_rx_waveform(tx_waveform, pulse, pulse_starts, pri_samples, PRIs, fs, fc, targets):
    total_samples = len(tx_waveform)
    rx_waveform = np.zeros(total_samples, dtype=complex)

    pulse_windows = []
    for i, (start, n_samples, pri_val) in enumerate(zip(pulse_starts, pri_samples, PRIs)):
        start_time = start / fs
        end_time = (start + n_samples) / fs
        pulse_windows.append({
            'start_time': start_time,
            'end_time': end_time,
            'start_sample': start,
            'pri_us': round(pri_val * 1e6, 1),
            'index': i
        })


    for rng, vel, rcs in targets:
        delay_time = 2 * rng / c
        doppler_shift = 2 * vel * fc / c

        for tx_idx, tx_sample in enumerate(pulse_starts):
            # tx_time = tx_sample / fs
            delay_samples = int(delay_time * fs)
            # pri_len = pri_samples[tx_idx]
              
            echo_sample = tx_sample + delay_samples
            if echo_sample + len(pulse) < total_samples:
                t = np.arange(len(pulse)) / fs + (echo_sample / fs)
                echo = rcs * pulse * np.exp(1j * 2 * np.pi * doppler_shift * t)
                rx_waveform[echo_sample:echo_sample + len(pulse)] += echo

    # Add noise
    rx_waveform += 10 * (np.random.randn(total_samples) + 1j * np.random.randn(total_samples))
    return rx_waveform, pulse_windows

# --- Build data cube ---
def build_data_cube(rx_waveform, pulse_starts, pri_samples, Nrange):
    data_cube = np.zeros((len(pulse_starts), Nrange), dtype=complex)
    for i, start in enumerate(pulse_starts):
        end = start + pri_samples[i]
        data_cube[i, :pri_samples[i]] = rx_waveform[start:end]
    return data_cube, Nrange

# --- Process range-Doppler per PRI ---
def process_range_doppler(data_cube, PRIs, pri_groups, pulse, fs, c, fc, max_samples):
    matched_filter = np.conj(np.fft.fft(pulse, n=max_samples))
    RD_maps = {}

    for pri_us in sorted(set(round(p * 1e6, 1) for p in PRIs)):
        indices = pri_groups[pri_us]
        group_data = data_cube[indices]
        doppler_N = len(indices)

        # Range compression
        compressed = np.array([
            np.fft.ifft(np.fft.fft(group_data[i, :]) * matched_filter)
            for i in range(doppler_N)
        ])

        # Doppler FFT
        fft_data = np.fft.fftshift(np.fft.fft(compressed, axis=0), axes=0)
        fft_db = 20 * np.log10(np.abs(fft_data) / np.mean(np.abs(fft_data)) + 1e-12)

        doppler_freq = np.fft.fftshift(np.fft.fftfreq(doppler_N, d=pri_us * 1e-6))
        doppler_velocity = doppler_freq * c / (2 * fc)
        range_vector = np.arange(max_samples) / fs * c / 2 / 1e3  # km

        RD_maps[pri_us] = {
            'fft_db': fft_db,
            'doppler_velocity': doppler_velocity,
            'range_vector': range_vector
        }

    return RD_maps

# --- Plot range-Doppler maps ---
def plot_rd_maps(RD_maps, DR, detections, pre_candidates):
    """
    Plots Range–Doppler maps for each PRI and overlays CFAR detections (and optional pre-detections).

    Parameters:
        RD_maps : dict {PRI_us: {'fft_db': ndarray}}
        DR : dynamic range (dB) for color scale
        detections : list of confirmed detections (from CFAR)
        pre_candidates : list of pre-detection candidates (optional)
    """
    fig, axs = plt.subplots(1, len(RD_maps), figsize=(20, 5), sharey=True)

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
        ax.set_xlabel("Doppler Bins (centered)")
        if i == 0:
            ax.set_ylabel("Range Bins")

    plt.tight_layout()
    plt.suptitle("Range–Doppler Maps with Detections", y=1.05, fontsize=16)
    plt.show()

# --- Detection ---
def pre_detection(RD_maps, pri_groups, fs, c, fc, TH1):
    """
    Detects strong peaks in each RD map.
    Returns a list of detected peaks with physical parameters.
    Each item: dict with PRI_us, doppler_bin, doppler_freq, velocity,
               range_bin, folded_range_km, power_dB
    """
    candidates = []
    for pri_us, data in RD_maps.items():
        fft_db = data['fft_db']
        peak_threshold = np.max(fft_db) - TH1

        for d_bin in range(fft_db.shape[0]):  # Doppler bins
            for r_bin in range(fft_db.shape[1]):  # Range bins
                power = fft_db[d_bin, r_bin]
                if power >= peak_threshold:
                    candidates.append({
                        'PRI_us': pri_us,
                        'doppler_bin': d_bin-fft_db.shape[0]/2,
                        'range_bin': r_bin,
                        'power_dB': power,
                    })

    return candidates

def cfar_detection(RD_maps, candidates, fs, c, fc, Tr, Td, Gr, Gd, offset_dB):
    """
    Candidate-based 2D Cell Averaging CFAR (CA-CFAR)

    Assumes each RDM in RD_maps has shape (Nd, Nr) = (doppler bins, range bins)

    Parameters
    ----------
    RD_maps : dict
        Dictionary of {PRI_us: {'fft_db': 2D array, ...}}
    candidates : list of dict
        Each dict must contain keys: 'PRI_us', 'range_bin', 'doppler_bin'
    fs, c, fc : float
        Radar parameters (unused here, for consistency)
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
        Nd, Nr = RDM_dB.shape  # Nd = doppler bins (rows), Nr = range bins (cols)

        # Candidate indices
        i_range = int(cand['range_bin'])                # range → column
        j_doppler = int((cand['doppler_bin'] + Nd//2) % Nd)  # doppler → row (fftshifted)

        # Skip edges
        if (i_range < Tr + Gr or i_range >= Nr - (Tr + Gr) or
            j_doppler < Td + Gd or j_doppler >= Nd - (Td + Gd)):
            continue

        # Linear scale
        RDM_linear = 10 ** (RDM_dB / 10)

        # Extract window around CUT
        window = RDM_linear[j_doppler - (Td + Gd): j_doppler + (Td + Gd) + 1,
                            i_range - (Tr + Gr): i_range + (Tr + Gr) + 1]

        # Guard mask (True = guard + CUT)
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
            folded_range = (i_range / fs) * c / 2  # meters
          
            cand_out = cand.copy()
            cand_out['doppler_freq'] = doppler_freq 
            cand_out['velocity'] = velocity
            cand_out['folded_range_km'] = folded_range / 1000
            cand_out['CFAR_threshold_dB'] = threshold_dB
            cand_out['CUT_dB'] = CUT_dB
            detections.append(cand_out)

    return detections



# --- Association ---
def group_detections_by_doppler_freq(detections, freq_tol_Hz=50):
    """
    Groups detected peaks by Doppler frequency (not bin index),
    merging close frequencies within freq_tol_Hz.
    """
    groups = defaultdict(list)

    for detection in detections:
        freq = detection['doppler_freq']
        # Round or quantize frequency to the nearest tolerance bin
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

def resolve_true_range(pris_us, folded_ranges_km, max_zones, tol_m):
    """
    Resolve true range for a target folded across multiple PRIs,
    allowing different zone index (m) per PRI.

    Parameters:
        pris_us: List of PRIs in microseconds.
        folded_ranges_km: List of measured folded ranges (in km), same length as pris_us.
        max_zones: Maximum number of zones to check for each PRI.
        tol_m: Tolerance in km for acceptable spread in computed R_true.

    Returns:
        dict with fields:
            'R_true'       : Estimated true range (km)
            'zone_indices' : List of m_i used for each PRI
            'r_true_list'  : Computed R_true per PRI
            'success'      : True if consistent R_true found, else False
    """
    N = len(pris_us)

    # Generate all possible m combinations (m_1, m_2, ..., m_N) from 0 to max_zones-1
    for m_combo in product(range(max_zones), repeat=N):
        r_true_list = []
        for i in range(N):
            pri_sec = pris_us[i] * 1e-6
            R_fold_km = folded_ranges_km[i]
            R_zone_km = m_combo[i] * (c * pri_sec / 2) * 1e-3  # zone offset in km
            r_true = R_zone_km + R_fold_km
            r_true_list.append(r_true)

        if np.ptp(r_true_list) < tol_m:
            return {
                'R_true': np.mean(r_true_list),
                'zone_indices': m_combo,
                'r_true_list': r_true_list,
                'success': True
            }

    # If no consistent R_true found
    return {
        'R_true': None,
        'zone_indices': None,
        'r_true_list': [],
        'success': False
    }

def unfold_multiple_detections(detections, max_zones, fc, tol_m):
    groups = group_detections_by_doppler_freq(detections)
    results = []

    for doppler_freq, peaks in groups.items():
        pris_us = [p['PRI_us'] for p in peaks]
        folded_ranges_km = [p['folded_range_km'] for p in peaks]

        result = resolve_true_range(pris_us, folded_ranges_km, max_zones, tol_m)
        if result['success']:
            results.append({
                'Velocity': (doppler_freq * c) / (2 * fc),
                'R_true': result['R_true'],
                'zone_indices': result['zone_indices'],
                'r_true_list': result['r_true_list'],
                'peaks': peaks
            })

    return results

# --- Main ---
def main():
    # Wveform Params
    PRI_set = [44.5, 40, 38, 34.5, 31]  # µs
    Npulse = 2048 # number of pulses (slow time)
    Nrange = 1024 # range samples (fast time)
    waveform = 'barker_13' # waveform type
    f_start = -10e6 # lfm start frequency
    f_end = 10e6 # lfm end frequency
    n_bin = 19 # number of bins
    fc = 34.5e9
    fs = 20e6
    
    # Detection and Association Params
    DR = 50  # dynamic range in dB
    max_zones = 12 # max zones for Arange
    tol_m = 0.1  # tolerance for Arange
    TH1 = 13 # pre detection threshold   
    TH2 = 30 # CFAR threshold    
    
    # Targets
    targets = define_targets()
    
    # Pulse Generator
    pulse = generate_pulse(fs, waveform, n_bin, f_start, f_end)
    
    RD_maps = {}
    
    for i, PRI in enumerate(PRI_set, start=1):
        print(f"\nProcessing PRI {i} = {PRI} µs")
        
        # TX waveform
        tx_waveform, pulse_starts, pri_samples, PRIs, total_samples = build_tx_waveform(pulse, [PRI], fs, Npulse)

        # RX signal with targets
        rx_waveform, pulse_windows = simulate_rx_waveform(tx_waveform, pulse, pulse_starts, pri_samples, PRIs, fs, fc, targets)
        
        # Data cube
        data_cube, max_samples = build_data_cube(rx_waveform, pulse_starts, pri_samples, Nrange)

        # PRI grouping
        pri_groups = defaultdict(list)
        for i, pri_val in enumerate(PRIs):
            pri_groups[round(pri_val * 1e6, 1)].append(i)
            
        # Process RD maps
        RD_map = process_range_doppler(data_cube, PRIs, pri_groups, pulse, fs, c, fc, max_samples)
        RD_maps.update(RD_map)
    
    # Detection
    candidates = pre_detection(RD_maps, pri_groups, fs, c, fc, TH1)
    print("\n--- Detected Pre Plots ---")
    print(tabulate(candidates, headers="keys", tablefmt="pretty", floatfmt=".2f")) 
    detections = cfar_detection(RD_maps, candidates, fs, c, fc, Tr=10, Td=12, Gr=6, Gd=3, offset_dB=TH2)
    print("\n--- Detected Plots ---")
    print(tabulate(detections, headers="keys", tablefmt="pretty", floatfmt=".2f"))   
    
    # Plot maps from 5 cycles
    plot_rd_maps(RD_maps, DR, detections, candidates)
    
    # Association
    unfold_results = unfold_multiple_detections(detections, max_zones, fc, tol_m)

    print("\n--- Associated Targets ---")
    for res in unfold_results:
        print(f"✅ Estimated range: {res['R_true']:.2f} km")
        print(f"✅ Estiamted velocity: {res['Velocity']:.2f} m/s")
        print(f"Used zone indices m: {res['zone_indices']}")
        print("PRI to folded range association:")
        for peak, r_true in zip(res['peaks'], res['r_true_list']):
            print(f"  PRI: {peak['PRI_us']} µs folded {peak['folded_range_km']:.2f} km -> R_true: {r_true:.3f} km")
        print("\n")
            
# Run
if __name__ == "__main__":
    main()
