"""
heart_rate.py
-------------
Module responsibility:
Extract Heart Rate (HR in Beats Per Minute - BPM) from filtered rPPG pulse signals
using Peak Detection and Spectral (FFT) analysis with exponential smoothing.
"""

import numpy as np
from scipy.signal import find_peaks

def calculate_heart_rate(
    filtered_signal: np.ndarray,
    fps: float = 30.0,
    prev_hr: float = None,
    smoothing_factor: float = 0.3
):
    """
    Extracts Heart Rate (BPM), peak positions, and signal quality metrics from a filtered pulse signal.

    Parameters:
        filtered_signal (ndarray): 1D bandpass-filtered pulse signal (~10 seconds window)
        fps (float): Sampling rate in Hz (Frames Per Second)
        prev_hr (float): Previous smoothed heart rate for temporal smoothing
        smoothing_factor (float): Weight for exponential smoothing (0.0 to 1.0)

    Returns:
        dict containing:
            - 'heart_rate': float (BPM)
            - 'raw_heart_rate': float (unsmoothed BPM)
            - 'number_of_peaks': int
            - 'peak_positions': ndarray of peak indices
            - 'signal_quality': float (0.0 to 1.0 quality index)
            - 'peak_intervals': list of valid inter-beat intervals in seconds
    """
    sig = np.array(filtered_signal, dtype=np.float64)

    # Return defaults if signal is empty or invalid
    if sig.ndim != 1 or len(sig) == 0 or np.std(sig) < 1e-6:
        return {
            'heart_rate': 0.0,
            'raw_heart_rate': 0.0,
            'number_of_peaks': 0,
            'peak_positions': np.array([]),
            'signal_quality': 0.0,
            'peak_intervals': []
        }

    num_samples = len(sig)

    # -------------------------------------------------------------------
    # 1. Peak Detection using scipy.signal.find_peaks
    # -------------------------------------------------------------------
    # Minimum peak distance corresponds to max human HR (220 BPM -> ~0.27s between peaks)
    min_dist = max(1, int(fps * 60.0 / 220.0))

    # Prominence threshold derived from signal standard deviation
    prominence = 0.2 * np.std(sig)

    peaks, properties = find_peaks(sig, distance=min_dist, prominence=prominence)

    # -------------------------------------------------------------------
    # 2. Inter-Beat Interval (IBI) Calculation & Impossible Interval Rejection
    # -------------------------------------------------------------------
    valid_intervals = []
    if len(peaks) > 1:
        raw_intervals = np.diff(peaks) / fps  # Convert sample intervals to seconds

        # Physiologically possible intervals: 0.27s (220 BPM) to 1.5s (40 BPM)
        for interval in raw_intervals:
            if 0.27 <= interval <= 1.50:
                valid_intervals.append(interval)

    # -------------------------------------------------------------------
    # 3. Peak-Based & FFT-Based HR Computation
    # -------------------------------------------------------------------
    raw_hr_peak = 0.0
    if len(valid_intervals) > 0:
        mean_ibi = np.mean(valid_intervals)
        raw_hr_peak = 60.0 / mean_ibi

    # FFT Spectral Peak Analysis for robust validation
    fft_spectrum = np.abs(np.fft.rfft(sig))
    fft_freqs = np.fft.rfftfreq(num_samples, d=1.0/fps)

    # Limit search frequency range to 0.7 Hz (42 BPM) to 3.5 Hz (210 BPM)
    valid_idx = np.where((fft_freqs >= 0.7) & (fft_freqs <= 3.5))[0]

    raw_hr_fft = 0.0
    signal_quality = 0.0

    if len(valid_idx) > 0:
        valid_spectrum = fft_spectrum[valid_idx]
        max_idx = valid_idx[np.argmax(valid_spectrum)]
        peak_freq = fft_freqs[max_idx]
        raw_hr_fft = peak_freq * 60.0

        # Calculate Signal Quality Index (Ratio of dominant peak power to surrounding noise)
        total_power = np.sum(valid_spectrum)
        peak_power = fft_spectrum[max_idx]
        if total_power > 0:
            signal_quality = float(np.clip(peak_power / total_power, 0.0, 1.0))

    # -------------------------------------------------------------------
    # 4. Combine HR Estimates & Apply Smoothing
    # -------------------------------------------------------------------
    if raw_hr_peak > 0 and raw_hr_fft > 0:
        # If peak-based HR and FFT-based HR are close (within 15 BPM), average them
        if abs(raw_hr_peak - raw_hr_fft) <= 15.0:
            raw_hr = 0.5 * (raw_hr_peak + raw_hr_fft)
        else:
            raw_hr = raw_hr_fft  # Prefer FFT for noisy signals
    elif raw_hr_fft > 0:
        raw_hr = raw_hr_fft
    elif raw_hr_peak > 0:
        raw_hr = raw_hr_peak
    else:
        raw_hr = 0.0

    # Apply Exponential Moving Average (EMA) smoothing to prevent wild jumps
    if prev_hr is not None and prev_hr > 0 and raw_hr > 0:
        smoothed_hr = smoothing_factor * raw_hr + (1.0 - smoothing_factor) * prev_hr
    else:
        smoothed_hr = raw_hr

    return {
        'heart_rate': round(float(smoothed_hr), 1),
        'raw_heart_rate': round(float(raw_hr), 1),
        'number_of_peaks': len(peaks),
        'peak_positions': peaks,
        'signal_quality': round(signal_quality, 2),
        'peak_intervals': valid_intervals
    }


# -----------------------------------------------------------------------
# Offline Synthetic Verification Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Running Heart Rate Extraction Offline Test...")

    fps = 60.0
    duration = 10.0  # 10 second window
    t = np.linspace(0, duration, int(fps * duration))

    # Known ground truth HR = 75 BPM -> 75 / 60 = 1.25 Hz
    true_bpm = 75.0
    freq = true_bpm / 60.0

    # Generate synthetic cardiac pulse wave (sine + 2nd harmonic)
    synthetic_pulse = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)

    result = calculate_heart_rate(synthetic_pulse, fps=fps)

    print(f"Ground Truth HR: {true_bpm} BPM")
    print(f"Extracted HR: {result['heart_rate']} BPM")
    print(f"Raw HR: {result['raw_heart_rate']} BPM")
    print(f"Number of Peaks Detected: {result['number_of_peaks']}")
    print(f"Signal Quality Index: {result['signal_quality']}")

    assert abs(result['heart_rate'] - true_bpm) <= 3.0, f"Error too large: expected {true_bpm}, got {result['heart_rate']}"
    print("[SUCCESS] Heart Rate extraction offline test passed perfectly!")
