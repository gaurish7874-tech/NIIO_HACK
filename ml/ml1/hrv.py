"""
hrv.py
------
Module responsibility:
Experimental Heart Rate Variability (HRV) estimation from consecutive heartbeat peak intervals.

DISCLAIMER:
This HRV module is EXPERIMENTAL and intended strictly for hackathon prototyping and wellness monitoring.
It is NOT a medical device and MUST NOT be used for clinical diagnosis or medical decision-making.
"""

import numpy as np

def calculate_hrv_rmssd(peak_positions, fps: float = 30.0):
    """
    Calculates the Root Mean Square of Successive Differences (RMSSD) of inter-beat intervals (RR intervals).

    Parameters:
        peak_positions (ndarray or list): Array of peak sample indices from video frames.
        fps (float): Sampling frequency in Hz (Frames Per Second).

    Returns:
        dict containing:
            - 'rmssd_ms': float (RMSSD in milliseconds, or None if insufficient valid intervals)
            - 'num_valid_intervals': int
            - 'mean_rr_ms': float (average RR interval in ms)
            - 'is_experimental': True
            - 'disclaimer': str
    """
    disclaimer_text = (
        "EXPERIMENTAL: For hackathon wellness demo only. Not clinically verified or medically accurate."
    )

    if peak_positions is None or len(peak_positions) < 3:
        return {
            'rmssd_ms': None,
            'num_valid_intervals': 0,
            'mean_rr_ms': None,
            'is_experimental': True,
            'disclaimer': disclaimer_text
        }

    peaks = np.sort(np.array(peak_positions, dtype=np.float64))

    # Convert peak frame differences to RR intervals in milliseconds
    # (frame_diff / fps) * 1000 = RR interval in ms
    raw_rr_ms = (np.diff(peaks) / fps) * 1000.0

    # -------------------------------------------------------------------
    # Interval Artifact Rejection
    # -------------------------------------------------------------------
    # Filter physiologically plausible RR intervals: 300 ms (200 BPM) to 1500 ms (40 BPM)
    valid_rr = []
    for rr in raw_rr_ms:
        if 300.0 <= rr <= 1500.0:
            valid_rr.append(rr)

    valid_rr = np.array(valid_rr)

    if len(valid_rr) < 2:
        return {
            'rmssd_ms': None,
            'num_valid_intervals': len(valid_rr),
            'mean_rr_ms': float(np.mean(valid_rr)) if len(valid_rr) > 0 else None,
            'is_experimental': True,
            'disclaimer': disclaimer_text
        }

    # -------------------------------------------------------------------
    # RMSSD Calculation
    # -------------------------------------------------------------------
    # Calculate successive differences between consecutive RR intervals
    rr_diffs = np.diff(valid_rr)

    # Filter out extreme artifact jumps between successive intervals (> 250 ms)
    clean_diffs = rr_diffs[np.abs(rr_diffs) <= 250.0]

    if len(clean_diffs) == 0:
        return {
            'rmssd_ms': None,
            'num_valid_intervals': len(valid_rr),
            'mean_rr_ms': float(np.mean(valid_rr)),
            'is_experimental': True,
            'disclaimer': disclaimer_text
        }

    # RMSSD = sqrt( mean( (RR_{i+1} - RR_i)^2 ) )
    rmssd = np.sqrt(np.mean(clean_diffs ** 2))

    return {
        'rmssd_ms': round(float(rmssd), 1),
        'num_valid_intervals': len(valid_rr),
        'mean_rr_ms': round(float(np.mean(valid_rr)), 1),
        'is_experimental': True,
        'disclaimer': disclaimer_text
    }


# -----------------------------------------------------------------------
# Offline Test with Synthetic Peak Data
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Running Experimental HRV Module Offline Test...")

    fps = 30.0
    # Simulate peaks corresponding to ~70 BPM with slight natural sinus arrhythmia (variation)
    # Target average RR interval = 60/70 = ~0.857 sec = ~25.7 frames at 30 FPS
    np.random.seed(42)
    synthetic_rr_sec = 0.857 + np.random.normal(loc=0.0, scale=0.03, size=15)  # +/- 30ms variation

    # Reconstruct frame indices
    frame_indices = np.cumsum(np.insert(synthetic_rr_sec * fps, 0, 10))

    result = calculate_hrv_rmssd(frame_indices, fps=fps)

    print(f"RMSSD (ms): {result['rmssd_ms']}")
    print(f"Valid Intervals: {result['num_valid_intervals']}")
    print(f"Mean RR Interval (ms): {result['mean_rr_ms']}")
    print(f"Experimental Status: {result['is_experimental']}")
    print(f"Disclaimer: {result['disclaimer']}")

    assert result['rmssd_ms'] is not None and result['rmssd_ms'] > 0
    print("[SUCCESS] Experimental HRV module test passed cleanly!")
