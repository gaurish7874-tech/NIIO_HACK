"""
pos.py
------
Module responsibility:
Plane-Orthogonal-to-Skin (POS) rPPG algorithm implementation using NumPy.

References:
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
Algorithmic Principles of Remote PPG. IEEE Transactions on Biomedical Engineering.
"""

import numpy as np

def extract_pos_pulse(rgb_signals, fps: float = 30.0) -> np.ndarray:
    """
    Extracts a 1D pulse waveform from a time-series of RGB spatial averages using the POS algorithm.

    Parameters:
        rgb_signals (ndarray or list): (N, 3) matrix of RGB values over N frames.
                                       Column 0: Red, Column 1: Green, Column 2: Blue.
        fps (float): Sampling frequency (Frames Per Second).

    Returns:
        pulse_signal (1D ndarray): Extracted 1D rPPG pulse signal of length N.
    """
    # -------------------------------------------------------------------
    # 1. Input Validation
    # -------------------------------------------------------------------
    rgb_arr = np.array(rgb_signals, dtype=np.float64)

    if rgb_arr.ndim != 2 or rgb_arr.shape[1] != 3:
        raise ValueError(f"Input rgb_signals must be a 2D array of shape (N, 3). Got shape {rgb_arr.shape}")

    num_frames = rgb_arr.shape[0]

    # Require minimum frames for statistical stability (at least ~1 second of data)
    min_frames = max(15, int(fps * 0.5))
    if num_frames < min_frames:
        print(f"[WARNING] Buffer contains only {num_frames} frames. Returning zeros.")
        return np.zeros(num_frames)

    # -------------------------------------------------------------------
    # 2. Temporal Normalization
    # Divide each channel by its mean across the window to eliminate illumination magnitude
    # -------------------------------------------------------------------
    mean_rgb = np.mean(rgb_arr, axis=0)

    # Avoid division by zero
    mean_rgb[mean_rgb == 0] = 1e-6

    normalized_rgb = rgb_arr / mean_rgb  # Shape: (N, 3)

    r_n = normalized_rgb[:, 0]
    g_n = normalized_rgb[:, 1]
    b_n = normalized_rgb[:, 2]

    # -------------------------------------------------------------------
    # 3. Projection onto Orthogonal Color Planes
    # S1 eliminates intensity specular variations; S2 provides orthogonal color contrast
    # -------------------------------------------------------------------
    s1 = g_n - b_n
    s2 = g_n + b_n - 2.0 * r_n

    # -------------------------------------------------------------------
    # 4. Standard Deviation Ratio (Alpha Tuning)
    # -------------------------------------------------------------------
    std_s1 = np.std(s1)
    std_s2 = np.std(s2)

    if std_s2 < 1e-6:
        alpha = 0.0
    else:
        alpha = std_s1 / std_s2

    # -------------------------------------------------------------------
    # 5. Combine Orthogonal Signals into 1D Pulse Waveform
    # -------------------------------------------------------------------
    pulse_signal = s1 + alpha * s2

    return pulse_signal


# -----------------------------------------------------------------------
# Offline Verification Test with Synthetic RGB Data
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Running POS Offline Synthetic Test...")

    fps = 30.0
    duration_seconds = 10.0
    t = np.linspace(0, duration_seconds, int(fps * duration_seconds))

    # Simulate 1.2 Hz (72 BPM) synthetic pulse signal
    pulse_freq = 1.2  # Hz
    true_pulse = 0.02 * np.sin(2 * np.pi * pulse_freq * t)

    # Simulate baseline skin RGB values + synthetic pulse absorption (Green channel absorbs most pulse)
    # Plus added illumination drift noise
    noise_drift = 2.0 * np.sin(2 * np.pi * 0.1 * t)

    r_channel = 180.0 + noise_drift + 0.2 * true_pulse
    g_channel = 120.0 + noise_drift + 1.0 * true_pulse  # Stronger pulse in Green
    b_channel = 90.0  + noise_drift + 0.1 * true_pulse

    synthetic_rgb = np.column_stack((r_channel, g_channel, b_channel))

    # Run POS algorithm
    pulse_output = extract_pos_pulse(synthetic_rgb, fps=fps)

    print(f"Synthetic RGB shape: {synthetic_rgb.shape}")
    print(f"Extracted pulse output shape: {pulse_output.shape}")
    print(f"Pulse Signal Mean: {np.mean(pulse_output):.6f}, Std: {np.std(pulse_output):.6f}")
    print("[SUCCESS] POS algorithm test executed cleanly!")
