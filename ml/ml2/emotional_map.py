"""
emotional_map.py
----------------
Arousal-Valence Emotional Mapping using Russell's Circumplex Model.
Maps combined physio + behavioral signals to a 2D emotional space.
"""

import logging

logger = logging.getLogger("ml2.emotional_map")

# Russell's Circumplex quadrant definitions
_QUADRANTS = {
    "stressed":    {"arousal": "high",  "valence": "negative", "label": "Stressed / Anxious",  "emoji": "high_neg"},
    "excited":     {"arousal": "high",  "valence": "positive", "label": "Excited / Alert",     "emoji": "high_pos"},
    "depressed":   {"arousal": "low",   "valence": "negative", "label": "Tired / Sad",         "emoji": "low_neg"},
    "relaxed":     {"arousal": "low",   "valence": "positive", "label": "Calm / Relaxed",      "emoji": "low_pos"},
    "neutral":     {"arousal": "mid",   "valence": "mid",      "label": "Neutral / Balanced",  "emoji": "neutral"},
}


def compute_arousal_valence(vitals: dict, baseline: dict = None) -> dict:
    """
    Maps physiological and behavioral signals to Russell's Circumplex Model.

    Parameters:
        vitals: The multimodal JSON from ML-1.
        baseline: Optional user baseline for personalized mapping.

    Returns:
        dict with:
            arousal (float): -1.0 (very low) to 1.0 (very high)
            valence (float): -1.0 (very negative) to 1.0 (very positive)
            quadrant (str): "stressed" | "excited" | "depressed" | "relaxed" | "neutral"
            quadrant_label (str): Human-readable label
            description (str): Brief explanation
    """
    physio = vitals.get("physio", {})
    behavioral = vitals.get("behavioral", {})

    # Baseline defaults
    b_hr = baseline.get("hr", 72.0) if baseline else 72.0
    b_sys = baseline.get("bp_sys", 120.0) if baseline else 120.0

    # -----------------------------------------------------------------------
    # AROUSAL: How activated/energized is the person?
    # High HR, high BP, fast respiration, high blink rate = high arousal
    # -----------------------------------------------------------------------
    hr = physio.get("hr") or b_hr
    sys_bp = physio.get("bp_sys") or b_sys
    resp = physio.get("respiration") or 15.0
    blink = behavioral.get("blink_rate", 17.0)
    eye_closure = behavioral.get("eye_closure", 0.1)

    # HR contribution: normalized around baseline
    hr_arousal = (hr - b_hr) / 40.0  # +1 at baseline+40, -1 at baseline-40

    # BP contribution: higher sys_bp = high arousal
    bp_arousal = (sys_bp - b_sys) / 40.0  # +1 when sys=160, -1 when sys=80

    # Respiration contribution
    resp_arousal = (resp - 15.0) / 10.0  # +1 at 25 RPM, -1 at 5 RPM

    # Blink contribution
    blink_arousal = (blink - 17.0) / 15.0  # elevated blink = higher arousal

    # Eye closure is inverse arousal (drowsy = low arousal). Relax the penalty.
    eye_arousal = -(eye_closure - 0.1) * 1.5  

    arousal = (
        hr_arousal * 0.40
        + bp_arousal * 0.30
        + resp_arousal * 0.10
        + blink_arousal * 0.10
        + eye_arousal * 0.10
    )
    arousal = max(-1.0, min(1.0, arousal))

    # -----------------------------------------------------------------------
    # VALENCE: Is the state positive or negative?
    # This is harder to determine from purely physiological signals.
    # We use proxies:
    #   Positive: stable gaze, normal BP, moderate HR, low eye closure
    #   Negative: unstable gaze, high BP, extreme HR, high eye closure
    # -----------------------------------------------------------------------
    head_yaw = abs(behavioral.get("head_pose", {}).get("yaw", 0))
    head_pitch = behavioral.get("head_pose", {}).get("pitch", 0)

    # BP health: closer to baseline = positive, deviating = negative.
    bp_valence = 1.0 - (abs(sys_bp - 120.0) / 40.0)  

    # Eye closure: open = positive, closed = negative.
    eye_valence = (0.25 - eye_closure) * 2.0

    valence = (
        bp_valence * 0.60
        + eye_valence * 0.40
    )
    valence = max(-1.0, min(1.0, valence))

    # -----------------------------------------------------------------------
    # QUADRANT CLASSIFICATION
    # -----------------------------------------------------------------------
    if abs(arousal) < 0.15 and abs(valence) < 0.15:
        quadrant = "neutral"
    elif arousal >= 0 and valence < 0:
        quadrant = "stressed"
    elif arousal >= 0 and valence >= 0:
        quadrant = "excited"
    elif arousal < 0 and valence < 0:
        quadrant = "depressed"
    else:
        quadrant = "relaxed"

    quad_info = _QUADRANTS[quadrant]

    # Generate description
    arousal_word = "high" if arousal > 0.3 else ("low" if arousal < -0.3 else "moderate")
    valence_word = "positive" if valence > 0.2 else ("negative" if valence < -0.2 else "neutral")
    description = (
        f"Your emotional state maps to {arousal_word} arousal with {valence_word} valence, "
        f"placing you in the '{quad_info['label']}' quadrant of the emotional circumplex."
    )

    logger.info(f"Emotional map: arousal={arousal:.2f}, valence={valence:.2f}, quadrant={quadrant}")

    return {
        "arousal": round(arousal, 3),
        "valence": round(valence, 3),
        "quadrant": quadrant,
        "quadrant_label": quad_info["label"],
        "description": description,
    }
