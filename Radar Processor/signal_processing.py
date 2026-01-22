# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:47:50 2026

@author: royi.b
Signal processing module for radar waveform generation and simulation.
"""
import numpy as np

def build_tx_waveform(pulse, PRI_set, fs, Npulse):
    """
    Build transmit waveform and PRI schedule.
    
    Parameters
    ----------
    pulse : ndarray
        Complex pulse waveform
    PRI_set : list
        List of PRIs in microseconds
    fs : float
        Sampling frequency [Hz]
    Npulse : int
        Number of pulses to transmit
        
    Returns
    -------
    tuple : (tx_waveform, pulse_starts, pri_samples, PRIs, total_samples)
    """
    PRIs = np.tile(np.array(PRI_set) * 1e-6, Npulse)
    pri_samples = [int(round(pri * fs)) for pri in PRIs]
    total_samples = sum(pri_samples)
    tx_waveform = np.zeros(total_samples, dtype=complex)
    pulse_starts = []

    cursor = 0
    for pri_len in pri_samples:
        tx_waveform[cursor:cursor + len(pulse)] = pulse
        pulse_starts.append(cursor)
        cursor += pri_len

    return tx_waveform, pulse_starts, pri_samples, PRIs, total_samples

def simulate_rx_waveform(tx_waveform, pulse, pulse_starts, pri_samples, PRIs, fs, fc, targets):
    """
    Simulate received waveform in FPGA-style RX windows (RX window = PRI minus pulse width).
    Handles targets beyond unambiguous range and partial overlap with RX windows.

    Parameters
    ----------
    tx_waveform : ndarray
        Full TX waveform (reference)
    pulse : ndarray
        Pulse waveform
    pulse_starts : list or ndarray
        Sample indices where TX pulses start
    pri_samples : list or ndarray
        Number of samples per PRI
    PRIs : list or ndarray
        PRI values in seconds
    fs : float
        Sampling frequency [Hz]
    fc : float
        Carrier frequency [Hz]
    targets : list of tuples
        Each target as (range_m, velocity_m_s, rcs)

    Returns
    -------
    rx_waveform : ndarray
        Concatenated RX waveform from all RX windows
    pulse_windows : list of dicts
        Metadata for each PRI (TX start, RX start/end, RX buffer indices)
    """
    c = 3e8
    PW = len(pulse)

    # --- Build RX windows metadata ---
    pulse_windows = []
    total_rx_samples = 0
    for i, (tx_start, pri_len, pri_val) in enumerate(zip(pulse_starts, pri_samples, PRIs)):
        rx_start = tx_start + PW
        rx_end   = tx_start + pri_len
        rx_len   = pri_len - PW
        if rx_len <= 0:
            raise ValueError("RX window <= 0")
        pulse_windows.append({
            'index': i,
            'tx_start': tx_start,
            'rx_start': rx_start,
            'rx_end': rx_end,
            'rx_len': rx_len,
            'rx_buffer_start': total_rx_samples,
            'pri_us': round(pri_val * 1e6, 1),
            'pri_samples': pri_len
        })
        total_rx_samples += rx_len

    # --- Initialize concatenated RX waveform buffer ---
    rx_waveform = np.zeros(total_rx_samples, dtype=complex)

    # --- Plant echoes for each target ---
    for rng, vel, rcs in targets:
        delay_time = 2 * rng / c
        delay_samples = int(np.round(delay_time * fs))
        doppler_shift = 2 * vel * fc / c

        for win in pulse_windows:
            pri_len = win['pri_samples']
            folded_delay = delay_samples % pri_len # Fold delay into PRI
            echo_sample_abs = win['tx_start'] + folded_delay
            echo_end_abs = echo_sample_abs + PW
            # RX window overlap
            echo_start = max(echo_sample_abs, win['rx_start'])
            echo_end   = min(echo_end_abs, win['rx_end'])
            n = echo_end - echo_start
            if n > 0:
                rx_idx = win['rx_buffer_start'] + (echo_start - win['rx_start'])
                t = np.arange(n)/fs + echo_start/fs
                echo = rcs * pulse[:n] * np.exp(1j * 2*np.pi * doppler_shift * t)
                rx_waveform[rx_idx:rx_idx+n] += echo
    
    # Add correlated clutter (zero Doppler)  
    clutter_power_dB = 10
    clutter_power = 10**(clutter_power_dB/10)
    for win in pulse_windows:
        rx_start = win['rx_buffer_start']
        rx_len   = win['rx_len']
        # One clutter realization per PRI
        clutter = (np.random.randn(rx_len) + 1j*np.random.randn(rx_len))
        clutter *= np.sqrt(clutter_power / np.mean(np.abs(clutter)**2))
        rx_waveform[rx_start:rx_start+rx_len] += clutter*np.exp(1j * np.random.normal(0, 0.2))
        
    # Add thermal noise (AWGN)   
    noise_power_dB = -60     # relative to echo amplitude=1
    noise_power = 10**(noise_power_dB/10)
    noise = (np.random.randn(len(rx_waveform)) + 1j*np.random.randn(len(rx_waveform))) * np.sqrt(noise_power/2)
    rx_waveform += noise 
    
    # ---- DEBUG rx_waveform
    # import matplotlib.pyplot as plt
    # # Matched filter (conjugate time-reversed pulse)
    # mf = np.conj(pulse[::-1])

    # # Plot match filter response for first PRI only (optional)
    # first_win = pulse_windows[0]
    # rx_segment = rx_waveform[first_win['rx_buffer_start']:first_win['rx_buffer_start'] + first_win['rx_len']]
    # mf_response = np.abs(np.fft.ifft(np.fft.fft(rx_segment, n=2*len(rx_segment)) * np.fft.fft(mf, n=2*len(rx_segment))))

    # plt.figure(figsize=(10,4))
    # plt.plot(mf_response)
    # plt.title("Matched Filter Response (First PRI RX Window)")
    # plt.xlabel("Sample")
    # plt.ylabel("Amplitude")
    # plt.grid(True)
    # plt.show()
    
    return rx_waveform, pulse_windows


def build_data_cube(rx_waveform, pulse_windows):
    """
    Build data cube from concatenated FPGA-style RX waveform buffer.

    Parameters
    ----------
    rx_waveform : ndarray
        Concatenated RX waveform from simulate_rx_waveform
    pulse_windows : list of dicts
        Metadata from simulate_rx_waveform
    debug : bool
        If True, prints debug info

    Returns
    -------
    data_cube : ndarray
        Shape (num_pulses, Nrange), ready for range FFT
    max_samples : int
        Maximum RX window length used
    """
    Nrange = max([w['rx_len'] for w in pulse_windows])
    num_pulses = len(pulse_windows)
    # Maximum RX window length
    rx_lengths = [w['rx_len'] for w in pulse_windows]
    max_rx_len = max(rx_lengths)
    # Clip to Nrange
    max_samples = min(max_rx_len, Nrange)

    # Allocate data cube
    data_cube = np.zeros((num_pulses, max_samples), dtype=complex)

    for i, win in enumerate(pulse_windows):
        buf_start = win['rx_buffer_start']
        rx_len = win['rx_len']
        rx_segment = rx_waveform[buf_start:buf_start + rx_len]

        # Clip/pad to Nrange
        L = min(len(rx_segment), Nrange)
        data_cube[i, :L] = rx_segment[:L]

        # ---- DEBUG data cube
        # echo_idx = np.where(np.abs(rx_segment) > 0)[0]
        # print(f"Pulse {i}: RX_len={rx_len}, n_samples in cube={L}, echo indices={echo_idx}")

    return data_cube, max_samples


def process_range_doppler(data_cube, PRIs, pri_groups, pulse, fs, c, fc, dec):
    """
    Process range-Doppler maps per PRI group.
    
    Parameters
    ----------
    data_cube : ndarray
        2D array of received data [pulses x samples]
    PRIs : ndarray
        PRI values in seconds
    pri_groups : dict
        Dictionary mapping PRI_us to pulse indices
    pulse : ndarray
        Reference pulse for matched filtering
    fs : float
        Sampling frequency [Hz]
    c : float
        Speed of light [m/s]
    fc : float
        Carrier frequency [Hz]
    dec : int
        Decimation factor
        
    Returns
    -------
    RD_maps : dict
        Dictionary of range-Doppler maps per PRI
    """
    RD_maps = {}
    
    for pri_us in sorted(set(round(p * 1e6, 1) for p in PRIs)):
        indices = pri_groups[pri_us]
        group_data = data_cube[indices]
        doppler_N = len(indices)
        
        # Decimate
        group_data_dec = group_data[:, ::dec]
        fs_dec = fs / dec
        
        # Matched filter at decimated rate
        mf_nfft = group_data_dec.shape[1]
        pulse_dec = pulse[::dec]
        # ---- IFFT MF
        # matched_filter = np.fft.fft(np.conj(pulse_dec[::-1]), n=mf_nfft)
        # ---- CONV MF
        matched_filter = np.conj(pulse_dec[::-1])
        
        # Range compression
        # ---- IFFT allocate range compression
        # compressed_dec = np.zeros((doppler_N, mf_nfft), dtype=complex)
        # ---- CONV allocate range compression
        compressed_dec = np.zeros((doppler_N, len(matched_filter)+len(group_data_dec[1, :])-1), dtype=complex)
        # compressed_dec = np.zeros((doppler_N, mf_nfft), dtype=complex) (depending on conv mode)
        for i in range(doppler_N):
            # ---- IFFT range compression
            # compressed_dec[i, :] = np.fft.ifft(np.fft.fft(group_data_dec[i, :]) * matched_filter)
            # ---- CONV range compression
            compressed_dec[i, :] = np.convolve(group_data_dec[i, :], matched_filter, mode='full')
        
        # Match filter alignment (depending on conv mode)
        compressed_dec = np.roll(compressed_dec, 1, axis=1) 
        
        # ---- DEBUG DATA CUBE FOR ELI
        # max_val = np.max(np.abs(group_data_dec))
        # if max_val == 0:
        #     scale = 1.0
        # else:
        #     scale = 2047 / max_val
       
        # group_data_dec_scaled = group_data_dec * scale
        # I = np.real(group_data_dec_scaled)
        # Q = np.imag(group_data_dec_scaled)
        # I_q = np.round(I).astype(np.int16)
        # Q_q = np.round(Q).astype(np.int16)
        # I_q = np.clip(I_q, -2048, 2047)
        # Q_q = np.clip(Q_q, -2048, 2047)
        
        # Doppler FFT
        fft_data = np.fft.fftshift(np.fft.fft(compressed_dec, axis=0), axes=0)
        fft_db = 20 * np.log10(np.abs(fft_data) / np.mean(np.abs(fft_data)) + 1e-12)
        
        doppler_freq = np.fft.fftshift(np.fft.fftfreq(doppler_N, d=pri_us * 1e-6))
        doppler_velocity = doppler_freq * c / (2 * fc)
        range_vector = np.arange(mf_nfft) / fs_dec * c / 2 / 1e3
        
        RD_maps[pri_us] = {
            'fft_db': fft_db,
            'doppler_velocity': doppler_velocity,
            'range_vector': range_vector
        }
        
        
        # ---- DEBUG RD_maps
        # import matplotlib.pyplot as plt

        # num_doppler_bins, num_range_bins = fft_db.shape
        
        # # Doppler bin axis centered around 0
        # doppler_bins = np.arange(-num_doppler_bins // 2, num_doppler_bins // 2)
        # range_bins = np.arange(num_range_bins)
        
        # plt.imshow(
        #     fft_db.T,
        #     aspect='auto',
        #     cmap='jet',
        #     origin='lower',
        #     vmin=np.max(fft_db) - 50,
        #     vmax=np.max(fft_db),
        #     extent=[doppler_bins[0], doppler_bins[-1],
        #             range_bins[0], range_bins[-1]]
        # )
     
    return RD_maps