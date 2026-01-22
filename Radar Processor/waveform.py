# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:46:59 2026

@author: royi.b
Waveform generation module for radar pulse creation.
"""
import numpy as np


def generate_pulse(fs, waveform, maxPWclks, f_start, f_end):
    """
    Generate Barker or LFM pulse waveform.

    Parameters
    ----------
    fs : float
        Sampling frequency [Hz]
    waveform : str
        Waveform type: 'barker', 'lfm', 'mls'
    maxPWclks : int
       Maximal Pulse length parameter (depends on design)
    f_start : float
        Start frequency for LFM [Hz]
    f_end : float
        End frequency for LFM [Hz]

    Returns
    -------
    pulse_complex : ndarray
        Complex baseband pulse waveform
    """
    match waveform.lower():
        case 'barker':
            # Generate Barker-13 pulse samples
            code = np.array([+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1])              
            nbins = maxPWclks//len(code) # 19
            rise_time = 4
            fall_time = 4
            
            pw = np.repeat(code, nbins)
            wf = np.concatenate([np.zeros(rise_time), pw, np.zeros(fall_time)])

            t_pulse = np.arange(len(wf)) / fs
            pulse_complex = wf * np.exp(1j * 2 * np.pi * 1e6 * t_pulse)
            return pulse_complex
                     
        case 'mls':
            # Generate MLS-63 waveform
            code = np.array([
                1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1,
                -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1,
                1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1,
                1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1
            ])           
            nbins = maxPWclks//len(code) # 4
            rise_time = 1
            fall_time = 2
            
            pw = np.repeat(code, nbins)
            wf = np.concatenate([np.zeros(rise_time), pw, np.zeros(fall_time)])

            t_pulse = np.arange(len(wf)) / fs
            pulse_complex = wf * np.exp(1j * 2 * np.pi * 1e6 * t_pulse)
            return pulse_complex
        
        case 'lfm':
            # Generate LFM waveform
            pulse_duration = maxPWclks / fs
            t = np.arange(0, pulse_duration, 1/fs)
            k = (f_end - f_start) / pulse_duration  # chirp rate
            phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
            pulse_complex = np.exp(1j * phase)
            return pulse_complex    
        
        case _:
            raise ValueError(f"Unsupported waveform: {waveform}")