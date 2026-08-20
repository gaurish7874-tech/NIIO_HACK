"""
filtering.py
------------
Module responsibility:
Bandpass filtering of POS pulse waveforms using SciPy signal processing.

Why Bandpass Filtering?
- Human heart rates physically range from ~42 BPM (0.7 Hz) to ~240 BPM (4.0 Hz).
- Low-frequency noise (< 0.7 Hz): Caused by body movement, respiration, and gradual room light changes.
- High-frequency noise (> 4.0 Hz): Caused by camera sensor noise and artificial lighting flicker.
- Bandpass filtering removes frequencies outside [0.7 Hz, 4.0 Hz], isolating clean cardiac pulses.
"""

import numpy as np
from scipy.signal import butter, filtfilt

def apply_bandpass_filter(
    signal: np.ndarray,
    fps: float = 30.0,
    lowcut: float = 0.7,
    highcut: float = 4.0,
    order: int = 3
) -> np.ndarray:
    """
    Applies a zero-phase Butterworth bandpass filter to a 1D pulse signal.

    Parameters:
        signal (ndarray): 1D input pulse waveform.
        fps (float): Sampling frequency in Hz (webcam FPS).
        lowcut (float): Minimum frequency in Hz (default: 0.7 Hz = ~42 BPM).
        highcut (float): Maximum frequency in Hz (default: 4.0 Hz = ~240 BPM).
        order (int): Filter order (default: 3).

    Returns:
        filtered_signal (ndarray): Filtered 1D pulse signal.
    """
    signal_arr = np.array(signal, dtype=np.float64)

    # Input validation
    if signal_arr.ndim != 1 or len(signal_arr) == 0:
        raise ValueError("Input signal must be a non-empty 1D numpy array.")

    # Require minimum signal length for filtering stability
    min_len = 3 * max(order, 1) + 1
    if len(signal_arr) < min_len:
        print(f"[WARNING] Signal length {len(signal_arr)} too short for filtering (min: {min_len}). Returning original signal.")
        return signal_arr

    # Ensure effective sampling rate is realistic (min 10.0 Hz to prevent low-FPS camera warmup crashes)
    effective_fps = max(fps, 10.0)
    nyquist = 0.5 * effective_fps

    # Safely clamp frequency cutoffs relative to Nyquist frequency
    safe_low = min(lowcut, nyquist - 0.2)
    safe_low = max(0.05, safe_low)

    safe_high = min(highcut, nyquist - 0.05)
    safe_high = max(safe_low + 0.1, safe_high)

    # Normalize frequencies relative to Nyquist (0.0 to 1.0)
    low = float(np.clip(safe_low / nyquist, 0.01, 0.98))
    high = float(np.clip(safe_high / nyquist, low + 0.01, 0.99))

    # Create Butterworth bandpass filter coefficients (b, a)
    b, a = butter(N=order, Wn=[low, high], btype='bandpass')

    # Apply zero-phase forward-backward filtering to prevent time shifts (phase distortion)
    filtered_signal = filtfilt(b, a, signal_arr)

    return filtered_signal


# -----------------------------------------------------------------------
# Offline Test with Visualization Plot (Matplotlib)
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    # Ensure project root is in Python path for standalone script execution
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    import matplotlib
    matplotlib.use('Agg')  # Headless backend
    import matplotlib.pyplot as plt
    from ml.ml1.pos import extract_pos_pulse

    print("Running POS + Bandpass Filter Offline Visual Test...")

    fps = 30.0
    duration = 10.0
    t = np.linspace(0, duration, int(fps * duration))

    # 1. Generate clean cardiac pulse at 1.2 Hz (72 BPM)
    clean_pulse = 0.03 * np.sin(2 * np.pi * 1.2 * t)

    # 2. Add low-frequency respiratory drift (0.15 Hz)
    low_freq_drift = 0.1 * np.sin(2 * np.pi * 0.15 * t)

    # 3. Add high-frequency camera noise (8.0 Hz)
    high_freq_noise = 0.02 * np.random.normal(size=len(t)) + 0.015 * np.sin(2 * np.pi * 8.0 * t)

    # Combine into raw noisy pulse signal
    raw_noisy_signal = clean_pulse + low_freq_drift + high_freq_noise

    # 4. Apply SciPy Bandpass Filter (0.7 Hz to 4.0 Hz)
    filtered_output = apply_bandpass_filter(raw_noisy_signal, fps=fps, lowcut=0.7, highcut=4.0)

    print(f"Raw Signal shape: {raw_noisy_signal.shape}")
    print(f"Filtered Signal shape: {filtered_output.shape}")

    # 5. Plot comparison graphs
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, raw_noisy_signal, color='red', alpha=0.7, label='Raw Noisy POS Signal')
    plt.title("rPPG Signal Processing: Raw vs Bandpass Filtered")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.plot(t, filtered_output, color='green', linewidth=2, label='Filtered Signal (0.7 - 4.0 Hz)')
    plt.plot(t, clean_pulse, color='blue', linestyle=':', alpha=0.8, label='Ground Truth 1.2 Hz Pulse (72 BPM)')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()

    plot_path = os.path.join(PROJECT_ROOT, "pos_bandpass_filter_test.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[SUCCESS] Filter test complete! Plot saved to: {plot_path}")
