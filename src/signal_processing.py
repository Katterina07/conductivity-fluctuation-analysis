"""
Signal processing utilities for conductivity fluctuation measurements.
"""

from pathlib import Path

import numpy as np
from scipy.signal import welch


def load_q00_signal(file_path: Path, sampling_rate: float):
    """
    Load a binary .Q00 file as a float32 time series.
    """
    signal = np.fromfile(file_path, dtype="<f4")
    time = np.arange(len(signal)) / sampling_rate

    return signal, time


def extract_spectrum_features(signal: np.ndarray, sampling_rate: float):
    """
    Estimate PSD and extract 1/f noise features.
    """
    frequencies, psd = welch(
        signal,
        fs=sampling_rate,
        nperseg=min(4096, len(signal)),
        detrend="constant",
    )

    mask = frequencies > 0
    frequencies = frequencies[mask]
    psd = psd[mask]

    log_frequencies = np.log10(frequencies)
    log_psd = np.log10(psd.clip(lower=1e-30))

    slope, intercept = np.polyfit(log_frequencies, log_psd, 1)

    return {
        "gamma": -slope,
        "energy": np.trapz(psd, frequencies),
        "frequencies": frequencies,
        "psd": psd,
    }
