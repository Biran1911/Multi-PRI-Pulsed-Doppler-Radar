import time
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
    # (range_m, vel_m_s, rcs)
    return [
        (5000, 0, 100),
        (11500, 30, 100),
        (35000, -25, 100),
        (40000, 10, 100),
    ]

# --- Generate baseband pulse ---
def generate_pulse(fs, waveform, nbin, f_start, f_end):
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
            pulse_duration = nbin / fs
            t = np.arange(0, pulse_duration, 1/fs)
            k = (f_end - f_start) / pulse_duration
            phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
            pulse_complex = np.exp(1j * phase)
            return pulse_complex
        case _:
            raise ValueError(f"Unsupported waveform: {waveform}")

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
            delay_samples = int(delay_time * fs)
            echo_sample = tx_sample + delay_samples
            if echo_sample + len(pulse) < total_samples:
                t = np.arange(len(pulse)) / fs + (echo_sample / fs)
                echo = rcs * pulse * np.exp(1j * 2 * np.pi * doppler_shift * t)
                rx_waveform[echo_sample:echo_sample + len(pulse)] += echo

    # Add noise (same scaling as original code)
    rx_waveform += 10 * (np.random.randn(total_samples) + 1j * np.random.randn(total_samples))
    return rx_waveform, pulse_windows

# --- Build data cube ---
def build_data_cube(rx_waveform, pulse_starts, pri_samples, Nrange):
    data_cube = np.zeros((len(pulse_starts), Nrange), dtype=complex)
    for i, start in enumerate(pulse_starts):
        end = start + pri_samples[i]
        # prevent out-of-bounds
        length = min(Nrange, end - start)
        data_cube[i, :length] = rx_waveform[start:start+length]
    return data_cube, Nrange

# --- Process range-Doppler per PRI ---
def process_range_doppler(data_cube, PRIs, pri_groups, pulse, fs, c, fc, max_samples):
    matched_filter = np.conj(np.fft.fft(pulse, n=max_samples))
    RD_maps = {}

    for pri_us in sorted(set(round(p * 1e6, 1) for p in PRIs)):
        indices = pri_groups[pri_us]
        group_data = data_cube[indices]
        doppler_N = len(indices)
        if doppler_N == 0:
            continue

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
    if not RD_maps:
        print("No RD maps to plot.")
        return

    fig, axs = plt.subplots(1, len(RD_maps), figsize=(20, 5), sharey=True)

    det_by_pri = {}
    if detections:
        for d in detections:
            det_by_pri.setdefault(d['PRI_us'], []).append(d)

    pre_by_pri = {}
    if pre_candidates:
        for p in pre_candidates:
            pre_by_pri.setdefault(p['PRI_us'], []).append(p)

    for i, (pri_us, data) in enumerate(sorted(RD_maps.items())):
        ax = axs[i] if len(RD_maps) > 1 else axs
        fft_db = data['fft_db']
        num_doppler_bins, num_range_bins = fft_db.shape

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

        if pri_us in pre_by_pri:
            for p in pre_by_pri[pri_us]:
                ax.plot(p['doppler_bin'],
                        p['range_bin'], 'yo', mfc='none', markersize=8, label='Pre-detect')

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

# --- Detection ---
def pre_detection(RD_maps, pri_groups, fs, c, fc, TH1):
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
    detections = []

    for cand in candidates:
        pri_us = cand['PRI_us']
        if pri_us not in RD_maps:
            continue

        RDM_dB = RD_maps[pri_us]['fft_db']
        Nd, Nr = RDM_dB.shape  # Nd = doppler bins (rows), Nr = range bins (cols)

        i_range = int(cand['range_bin'])                # range → column
        j_doppler = int((cand['doppler_bin'] + Nd//2) % Nd)  # doppler → row (fftshifted)

        if (i_range < Tr + Gr or i_range >= Nr - (Tr + Gr) or
            j_doppler < Td + Gd or j_doppler >= Nd - (Td + Gd)):
            continue

        RDM_linear = 10 ** (RDM_dB / 10)

        window = RDM_linear[j_doppler - (Td + Gd): j_doppler + (Td + Gd) + 1,
                            i_range - (Tr + Gr): i_range + (Tr + Gr) + 1]

        guard_mask = np.zeros_like(window, dtype=bool)
        guard_mask[Td:Td + 2*Gd + 1, Tr:Tr + 2*Gr + 1] = True

        training_cells = window[~guard_mask]
        if len(training_cells) == 0:
            continue

        noise_level = np.mean(training_cells)
        threshold_dB = 10 * np.log10(noise_level) + offset_dB
        CUT_dB = RDM_dB[j_doppler, i_range]

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
    groups = defaultdict(list)

    for detection in detections:
        freq = detection['doppler_freq']
        freq_key = round(freq / freq_tol_Hz) * freq_tol_Hz
        groups[freq_key].append(detection)

    reduced_groups = defaultdict(list)
    for freq_key, peaks in groups.items():
        by_pri = defaultdict(list)
        for p in peaks:
            by_pri[p['PRI_us']].append(p)
        for pri, plist in by_pri.items():
            best = max(plist, key=lambda x: x['power_dB'])
            reduced_groups[freq_key].append(best)
            
    return reduced_groups

def resolve_true_range(pris_us, folded_ranges_km, max_zones, tol_km):
    N = len(pris_us)

    for m_combo in product(range(max_zones), repeat=N):
        r_true_list = []
        for i in range(N):
            pri_sec = pris_us[i] * 1e-6
            R_fold_km = folded_ranges_km[i]
            R_zone_km = m_combo[i] * (c * pri_sec / 2) * 1e-3  # zone offset in km
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

# --- Main time-driven simulation (cycles PRIs 1..5, 2048 pulses per PRI) ---
def main():
    # --- Use your parameters exactly ---
    PRI_set = [44.5, 40, 38, 34.5, 31]  # µs
    Npulse = 2048
    Nrange = 1024
    waveform = 'barker_13'
    f_start = -10e6
    f_end = 10e6
    n_bin = 19
    fc = 34.5e9
    fs = 20e6

    DR = 50
    max_zones = 9
    tol_km = 0.1
    TH1 = 13
    TH2 = 30

    # Simulation time control (change sim_time as needed)
    sim_time = 10  # seconds (keeps runtime modest); change to 10.0 for longer sim
    sim_clock = 0.0

    # Initialize
    targets_init = define_targets()  # initial R0 in meters
    pulse = generate_pulse(fs, waveform, n_bin, f_start, f_end)

    # Storage for per-PRI RD maps and per-PRI detections within a PRI-cycle of 5 PRIs
    stored_RD_maps = {}       # {PRI_us: RD_map} for current cycle (max 5 entries)
    stored_candidates = []    # list of candidate dicts from each PRI in the cycle
    stored_detections = []    # list of detection dicts from each PRI in the cycle

    all_associations = []     # list of association results per completed 5-PRI cycle

    pri_idx = 0
    cycle_count = 0

    print(f"Starting time-driven sim: sim_time={sim_time}s, Npulse per PRI={Npulse}")

    # We'll loop until sim_clock reaches sim_time
    while sim_clock < sim_time:
        PRI = PRI_set[pri_idx % len(PRI_set)]
        PRI_us_rounded = round(PRI, 1)
        pri_idx += 1

        # Update targets according to current sim time (R(t) = R0 + v * t)
        targets = []
        for R0, v, rcs in targets_init:
            R_t = R0 + v * sim_clock
            targets.append((R_t, v, rcs))

        print(f"\n[sim t={sim_clock:.6f}s] PRI index {(pri_idx-1)%len(PRI_set)+1} -> {PRI} µs : transmitting {Npulse} pulses")

        t0 = time.time()
        # Transmit/Build TX for this PRI (Npulse pulses of same PRI)
        tx_waveform, pulse_starts, pri_samples, PRIs, total_samples = build_tx_waveform(pulse, [PRI], fs, Npulse)
        t_build = time.time() - t0

        t0 = time.time()
        # Simulate RX for this PRI (targets updated for this sim time)
        rx_waveform, pulse_windows = simulate_rx_waveform(tx_waveform, pulse, pulse_starts, pri_samples, PRIs, fs, fc, targets)
        t_simrx = time.time() - t0

        t0 = time.time()
        # Build data cube and process RD for this PRI block
        data_cube, max_samples = build_data_cube(rx_waveform, pulse_starts, pri_samples, Nrange)

        # Group pulses for this PRI (should be one PRI_us key)
        pri_groups = defaultdict(list)
        for ii, pri_val in enumerate(PRIs):
            pri_groups[round(pri_val * 1e6, 1)].append(ii)

        # Process RD maps for this PRI block
        RD_map = process_range_doppler(data_cube, PRIs, pri_groups, pulse, fs, c, fc, max_samples)
        t_proc = time.time() - t0

        # Per-PRI detection (pre + CFAR) using only RD_map for this PRI
        # pre_detection expects RD_maps dict, so we pass RD_map
        candidates = pre_detection(RD_map, pri_groups, fs, c, fc, TH1)
        detections = cfar_detection(RD_map, candidates, fs, c, fc, Tr=10, Td=12, Gr=6, Gd=3, offset_dB=TH2)

        # Print short summary for this PRI
        print(f"timings (s): build_tx={t_build:.3f}, sim_rx={t_simrx:.3f}, proc_rd={t_proc:.3f}")
        print(f" -> candidates: {len(candidates)}, detections: {len(detections)}")

        # Store RD_map and detection/candidates for association step
        # RD_map keys are PRI_us floats (one key like 31.0). We'll keep them.
        stored_RD_maps.update(RD_map)
        stored_candidates.extend(candidates)
        stored_detections.extend(detections)

        # Advance simulation clock by the total elapsed on-air time for the Npulse pulses of this PRI:
        # Each pulse period = PRI * 1e-6 (s). Total transmit duration = PRI * Npulse * 1e-6
        transmit_duration = PRI * 1e-6 * Npulse
        # Add small processing overhead (optional) to simulate processing time
        processing_overhead = 0.0  # keep 0; you can set e.g. 0.01 * transmit_duration if desired
        sim_clock += transmit_duration + processing_overhead

        # Every time we've done 5 distinct PRIs (i.e., completed a 1..5 set) -> association
        # We'll perform association when we have at least the 5 unique PRI keys present.
        if len(stored_RD_maps) >= len(PRI_set):
            cycle_count += 1
            print(f"\n>>> Performing association on cycle {cycle_count} at sim t={sim_clock:.6f}s")
            # Use all stored_detections (they have 'PRI_us' fields) to attempt unfolding
            assoc_results = unfold_multiple_detections(stored_detections, max_zones, fc, tol_km)

            # Print association summary
            if len(assoc_results) == 0:
                print("No associations found this cycle.")
            else:
                print(f"Associations found: {len(assoc_results)}")
                for res in assoc_results:
                    print(f"  -> R_true {res['R_true']:.3f} km, Vel {res['Velocity']:.2f} m/s, zones {res['zone_indices']}")

            # Save association results
            all_associations.append({
                'sim_time': sim_clock,
                'cycle': cycle_count,
                'assoc': assoc_results,
                'detections': stored_detections.copy()
            })

            # Clear stored RD maps and detections for the next 5-PRI cycle
            stored_RD_maps.clear()
            stored_candidates.clear()
            stored_detections.clear()

        # advance PRI index (cycle automatically in next while iteration)
        # loop will continue until sim_clock >= sim_time

    # End simulation: optionally plot final RD maps (if any)
    if stored_RD_maps:
        print("\nPlotting last stored RD maps (partial cycle)")
        plot_rd_maps(stored_RD_maps, DR, stored_detections, stored_candidates)

    print("\n--- Simulation finished ---")
    print(f"Total association cycles: {len(all_associations)}")
    for a in all_associations:
        print(f" cycle {a['cycle']} @ t={a['sim_time']:.6f}s -> {len(a['assoc'])} associations")

    return all_associations

if __name__ == "__main__":
    main()
