"""
wellness_score.py
-----------------
Computes a composite Wellness Score (0-100) from raw vitals.
0 = critical, 100 = thriving. Acts as the instant "hook" for demos.
"""

import logging

logger = logging.getLogger("ml2.wellness_score")


def _normalize_hr(hr: float, baseline_hr: float = 72.0) -> float:
    """Score heart rate: 1.0 at ideal, 0.0 at extremes."""
    if hr is None:
        return 0.5  # Unknown = neutral
    deviation = abs(hr - baseline_hr)
    return max(0.0, 1.0 - (deviation / 50.0))


def _normalize_hrv(hrv: float, baseline_hrv: float = 50.0) -> float:
    """Score HRV: higher = better. 1.0 at 60+ms, 0.0 at 0ms."""
    if hrv is None:
        return 0.5
    return min(hrv / max(baseline_hrv * 1.2, 60.0), 1.0)


def _normalize_respiration(resp: float, baseline_resp: float = 15.0) -> float:
    """Score respiration: 1.0 at ideal (12-18), 0.0 at extremes."""
    if resp is None:
        return 0.5
    deviation = abs(resp - baseline_resp)
    return max(0.0, 1.0 - (deviation / 12.0))


def _normalize_blink(blink_rate: float, baseline_blink: float = 17.0) -> float:
    """Score blink rate: 1.0 at normal (15-20), 0.0 at extremes."""
    deviation = abs(blink_rate - baseline_blink)
    return max(0.0, 1.0 - (deviation / 20.0))


def _normalize_gaze(gaze_stability: float) -> float:
    """Score gaze stability: direct passthrough (already 0-1)."""
    return max(0.0, min(1.0, gaze_stability))


def _normalize_eye_closure(eye_closure: float) -> float:
    """Score eye closure: 1.0 = open (good), 0.0 = closed (bad)."""
    return max(0.0, 1.0 - eye_closure)


def compute_wellness_score(vitals: dict, baseline: dict = None) -> dict:
    """
    Computes a composite Wellness Score (0-100).

    Parameters:
        vitals: The multimodal JSON from ML-1.
        baseline: Optional user baseline from calibration.
                  Keys: hr, hrv, respiration, blink_rate

    Returns:
        dict with:
            score (int): 0-100 composite wellness score
            category (str): "excellent" | "good" | "fair" | "poor" | "critical"
            breakdown (dict): per-signal scores for detailed view
            color (str): hex color for UI rendering
    """
    physio = vitals.get("physio", {})
    behavioral = vitals.get("behavioral", {})

    # Use baseline if available, otherwise use population defaults
    b_hr = baseline.get("hr", 72.0) if baseline else 72.0
    b_hrv = baseline.get("hrv", 50.0) if baseline else 50.0
    b_resp = baseline.get("respiration", 15.0) if baseline else 15.0
    b_blink = baseline.get("blink_rate", 17.0) if baseline else 17.0

    # Calculate individual signal scores (0.0 to 1.0)
    hr_score = _normalize_hr(physio.get("hr"), b_hr)
    hrv_score = _normalize_hrv(physio.get("hrv"), b_hrv)
    resp_score = _normalize_respiration(physio.get("respiration"), b_resp)
    blink_score = _normalize_blink(behavioral.get("blink_rate", 17.0), b_blink)
    gaze_score = _normalize_gaze(behavioral.get("gaze_stability", 0.5))
    eye_score = _normalize_eye_closure(behavioral.get("eye_closure", 0.0))

    # Factor in signal confidence
    physio_conf = physio.get("confidence", 0.5)
    behav_conf = behavioral.get("confidence", 0.5)

    # Weighted composite (physio slightly more important for wellness)
    weights = {
        "heart_rate": 0.22,
        "hrv": 0.18,
        "respiration": 0.10,
        "blink_rate": 0.10,
        "gaze_stability": 0.15,
        "eye_openness": 0.15,
        "signal_quality": 0.10,
    }

    signal_quality = (physio_conf + behav_conf) / 2.0

    raw_score = (
        hr_score * weights["heart_rate"]
        + hrv_score * weights["hrv"]
        + resp_score * weights["respiration"]
        + blink_score * weights["blink_rate"]
        + gaze_score * weights["gaze_stability"]
        + eye_score * weights["eye_openness"]
        + signal_quality * weights["signal_quality"]
    )

    # Scale to 0-100
    score = int(round(raw_score * 100))
    score = max(0, min(100, score))

    # Categorize
    if score >= 85:
        category = "excellent"
        color = "#22c55e"  # green
    elif score >= 70:
        category = "good"
        color = "#84cc16"  # lime
    elif score >= 50:
        category = "fair"
        color = "#eab308"  # yellow
    elif score >= 30:
        category = "poor"
        color = "#f97316"  # orange
    else:
        category = "critical"
        color = "#ef4444"  # red

    breakdown = {
        "heart_rate": round(hr_score, 2),
        "hrv": round(hrv_score, 2),
        "respiration": round(resp_score, 2),
        "blink_rate": round(blink_score, 2),
        "gaze_stability": round(gaze_score, 2),
        "eye_openness": round(eye_score, 2),
        "signal_quality": round(signal_quality, 2),
    }

    logger.info(f"Wellness Score: {score}/100 ({category})")
    return {
        "score": score,
        "category": category,
        "color": color,
        "breakdown": breakdown,
    }
