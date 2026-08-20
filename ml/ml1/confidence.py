"""
confidence.py
-------------
Module responsibility:
Calculate dynamic confidence score (0.0 to 1.0) for physiological measurement stability.

Why Confidence Scoring?
If a user is moving rapidly, in a dark room, or partially out of frame, the extracted HR might be noisy.
Rather than presenting a wrong HR with high certainty, the confidence score drops (e.g. 0.31),
notifying downstream systems (and ML-2 Triage) that the reading is unreliable.
"""

import numpy as np

def calculate_confidence_score(
    face_detected: bool,
    roi_valid: bool,
    buffer_length: int,
    signal_quality: float,
    peak_intervals: list = None,
    max_buffer_size: int = 300
) -> float:
    """
    Computes an aggregate confidence score between 0.0 (unusable) and 1.0 (high quality).

    Parameters:
        face_detected (bool): Whether a face landmark detection is active
        roi_valid (bool): Whether forehead skin ROI was extracted cleanly
        buffer_length (int): Current number of samples in the rolling buffer
        signal_quality (float): Signal SNR / spectral peak prominence ratio (0.0 to 1.0)
        peak_intervals (list): List of inter-beat intervals in seconds
        max_buffer_size (int): Max target buffer size (default 300)

    Returns:
        confidence (float): Score rounded to 2 decimal places (0.00 to 1.00)
    """
    # Hard zero if face or ROI is invalid
    if not face_detected or not roi_valid:
        return 0.0

    # 1. Buffer Fullness (30% weight)
    # Require at least ~90 frames (3 seconds) for initial confidence buildup
    buffer_ratio = min(1.0, buffer_length / max_buffer_size)
    if buffer_length < 90:
        buffer_weight = 0.15 * (buffer_length / 90.0)
    else:
        buffer_weight = 0.30 * buffer_ratio

    # 2. Signal Quality Index from Spectral SNR (40% weight)
    quality_weight = 0.40 * min(1.0, max(0.0, signal_quality))

    # 3. Peak Consistency / Interval Regularity (30% weight)
    consistency_weight = 0.0
    if peak_intervals is not None and len(peak_intervals) >= 2:
        std_interval = np.std(peak_intervals)
        mean_interval = np.mean(peak_intervals)

        if mean_interval > 0:
            # Coefficient of Variation (CV) = std / mean
            cv = std_interval / mean_interval
            # High consistency = low CV (< 0.15)
            consistency_score = max(0.0, 1.0 - (cv / 0.30))
            consistency_weight = 0.30 * min(1.0, consistency_score)
    else:
        # Default baseline weight if buffer is building up
        consistency_weight = 0.15 * buffer_ratio

    total_score = buffer_weight + quality_weight + consistency_weight
    return round(float(np.clip(total_score, 0.0, 1.0)), 2)


# -----------------------------------------------------------------------
# Offline Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Running Confidence Score Module Offline Test...")

    # Case 1: Ideal clear conditions
    score_ideal = calculate_confidence_score(
        face_detected=True,
        roi_valid=True,
        buffer_length=300,
        signal_quality=0.85,
        peak_intervals=[0.8, 0.81, 0.79, 0.80, 0.82]
    )
    print(f"Ideal Conditions Confidence Score: {score_ideal}")

    # Case 2: Motion / Dark room / Low quality
    score_poor = calculate_confidence_score(
        face_detected=True,
        roi_valid=True,
        buffer_length=120,
        signal_quality=0.20,
        peak_intervals=[0.5, 1.1, 0.4, 0.9]  # Irregular peaks
    )
    print(f"Poor Conditions Confidence Score: {score_poor}")

    # Case 3: No face detected
    score_noface = calculate_confidence_score(
        face_detected=False,
        roi_valid=False,
        buffer_length=300,
        signal_quality=0.9
    )
    print(f"No Face Confidence Score: {score_noface}")

    assert score_ideal >= 0.70
    assert score_poor <= 0.45
    assert score_noface == 0.0
    print("[SUCCESS] Confidence score module test passed!")
