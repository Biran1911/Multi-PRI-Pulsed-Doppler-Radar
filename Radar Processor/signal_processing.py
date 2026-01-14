# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:47:50 2026

@author: royi.b
Signal processing module for radar waveform generation and simulation.
"""
import numpy as np
from collections import defaultdict


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
    Simulate received waveform with injected echoes from targets.
    
    Parameters
    ----------
    tx_waveform : ndarray
        Transmit waveform
    pulse : ndarray
        Pulse waveform
    pulse_starts : list
        Sample indices where pulses start
    pri_samples : list
        Number of samples per PRI
    PRIs : ndarray
        PRI values in seconds
    fs : float
        Sampling frequency [Hz]
    fc : float
        Carrier frequency [Hz]
    targets : list
        List of (range, velocity, rcs) tuples
        
    Returns
    -------
    tuple : (rx_waveform, pulse_windows)
    """
    c = 3e8
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

    return rx_waveform, pulse_windows


def build_data_cube(rx_waveform, pulse_starts, pri_samples, Nrange, dec):
    """
    Build data cube from received waveform.
    
    Parameters
    ----------
    rx_waveform : ndarray
        Received waveform
    pulse_starts : list
        Sample indices where pulses start
    pri_samples : list
        Number of samples per PRI
    Nrange : int
        Number of range samples
    dec : int
        Decimation factor
        
    Returns
    -------
    tuple : (data_cube, max_samples)
    """
    data_cube = np.zeros((len(pulse_starts), max(pri_samples)), dtype=complex)
    
    for i, start in enumerate(pulse_starts):
        end = start + pri_samples[i]
        data_cube[i, :pri_samples[i]] = rx_waveform[start:end]
        
    return data_cube, max(pri_samples)


def process_range_doppler(data_cube, PRIs, pri_groups, pulse, fs, c, fc, Nrange, dec):
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
    Nrange : int
        Number of range samples
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
        matched_filter = np.fft.fft(np.conj(pulse_dec[::-1]), n=mf_nfft)
        
        # Range compression
        compressed_dec = np.zeros((doppler_N, mf_nfft), dtype=complex)
        for i in range(doppler_N):
            compressed_dec[i, :] = np.fft.ifft(
                np.fft.fft(group_data_dec[i, :]) * matched_filter
            )
        compressed_dec = np.roll(compressed_dec, -(len(pulse_dec)-1), axis=1)
        
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
     
    return RD_maps