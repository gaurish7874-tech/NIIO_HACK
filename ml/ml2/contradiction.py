"""
contradiction.py
----------------
Cross-Modal Contradiction Detection.
Detects when physiological and behavioral signals tell conflicting stories
and generates an intelligent explanation of the discrepancy.
"""

import logging

logger = logging.getLogger("ml2.contradiction")

# Signal-to-state mapping for comparison
_STRESS_INDICATORS_PHYSIO = {
    "hr_high": lambda p: (p.get("hr") or 72) > 95,
    "hrv_low": lambda p: (p.get("hrv") or 50) < 25,
    "resp_high": lambda p: (p.get("respiration") or 15) > 20,
}

_CALM_INDICATORS_PHYSIO = {
    "hr_normal": lambda p: 55 <= (p.get("hr") or 72) <= 85,
    "hrv_good": lambda p: (p.get("hrv") or 50) > 40,
    "resp_normal": lambda p: 12 <= (p.get("respiration") or 15) <= 18,
}

_STRESS_INDICATORS_BEHAV = {
    "blink_high": lambda b: b.get("blink_rate", 17) > 25,
    "gaze_unstable": lambda b: b.get("gaze_stability", 0.7) < 0.4,
    "restless_head": lambda b: abs(b.get("head_pose", {}).get("yaw", 0)) > 12,
}

_CALM_INDICATORS_BEHAV = {
    "blink_normal": lambda b: 12 <= b.get("blink_rate", 17) <= 22,
    "gaze_stable": lambda b: b.get("gaze_stability", 0.7) > 0.6,
    "head_steady": lambda b: abs(b.get("head_pose", {}).get("yaw", 0)) < 8,
}

_FATIGUE_INDICATORS_BEHAV = {
    "eye_closure_high": lambda b: b.get("eye_closure", 0) > 0.35,
    "blink_low": lambda b: b.get("blink_rate", 17) < 10,
    "head_drooping": lambda b: b.get("head_pose", {}).get("pitch", 0) > 12,
}

# Explanation templates
_CONTRADICTION_TEMPLATES = {
    "physio_stressed_behav_calm": {
        "label": "Internal Stress, External Composure",
        "explanation": (
            "Your physiological signals (elevated heart rate, low HRV) suggest internal stress, "
            "but your behavioral signals (steady gaze, normal blink rate) show you are externally composed. "
            "This pattern is common in experienced professionals managing pressure. "
            "While you appear calm outwardly, your body is working hard. "
            "Consider taking a proactive break before physical symptoms escalate."
        ),
        "severity": "moderate",
    },
    "physio_calm_behav_stressed": {
        "label": "Restless but Physically Calm",
        "explanation": (
            "Your physiological signals (normal heart rate, healthy HRV) suggest your body is relaxed, "
            "but your behavioral signals (erratic gaze, elevated blink rate) indicate mental restlessness or distraction. "
            "This could mean environmental distractions, boredom, or early-stage anxiety that hasn't yet "
            "triggered a physical stress response. Consider refocusing with a brief mindfulness exercise."
        ),
        "severity": "mild",
    },
    "physio_stressed_behav_fatigued": {
        "label": "Stressed but Drowsy (Burnout Pattern)",
        "explanation": (
            "Your body shows stress signals (elevated HR, low HRV) while your behavioral signals show fatigue "
            "(heavy eyelids, slow blinks, head drooping). This combination — being both wired and tired — "
            "is a classic burnout pattern. Your body is in fight-or-flight mode but is too exhausted to sustain it. "
            "This is a strong signal to stop, rest, and recover. Pushing through will likely worsen both conditions."
        ),
        "severity": "high",
    },
    "physio_calm_behav_fatigued": {
        "label": "Natural Drowsiness",
        "explanation": (
            "Your physiological signals are calm and your behavioral signals show drowsiness. "
            "This is a natural fatigue pattern — your body and behavior are aligned in signaling "
            "that you need rest. This is not a stress response."
        ),
        "severity": "mild",
    },
}


def _count_matching(indicators: dict, data: dict) -> int:
    """Count how many indicator functions return True for the given data."""
    return sum(1 for fn in indicators.values() if fn(data))


def detect_contradiction(vitals: dict) -> dict:
    """
    Analyzes physio vs behavioral signals for contradictions.

    Parameters:
        vitals: The multimodal JSON from ML-1.

    Returns:
        dict with:
            has_contradiction (bool): Whether a meaningful contradiction was detected
            contradiction_type (str): Key identifying the contradiction pattern
            label (str): Short human-readable label
            explanation (str): Detailed explanation for the Judge/user
            severity (str): "mild" | "moderate" | "high"
    """
    physio = vitals.get("physio", {})
    behavioral = vitals.get("behavioral", {})

    # Score each modality for each state
    physio_stress_count = _count_matching(_STRESS_INDICATORS_PHYSIO, physio)
    physio_calm_count = _count_matching(_CALM_INDICATORS_PHYSIO, physio)
    behav_stress_count = _count_matching(_STRESS_INDICATORS_BEHAV, behavioral)
    behav_calm_count = _count_matching(_CALM_INDICATORS_BEHAV, behavioral)
    behav_fatigue_count = _count_matching(_FATIGUE_INDICATORS_BEHAV, behavioral)

    physio_stressed = physio_stress_count >= 2
    physio_calm = physio_calm_count >= 2
    behav_stressed = behav_stress_count >= 2
    behav_calm = behav_calm_count >= 2
    behav_fatigued = behav_fatigue_count >= 2

    # Detect contradiction patterns
    contradiction_type = None

    if physio_stressed and behav_calm:
        contradiction_type = "physio_stressed_behav_calm"
    elif physio_calm and behav_stressed:
        contradiction_type = "physio_calm_behav_stressed"
    elif physio_stressed and behav_fatigued:
        contradiction_type = "physio_stressed_behav_fatigued"
    elif physio_calm and behav_fatigued:
        contradiction_type = "physio_calm_behav_fatigued"

    if contradiction_type:
        template = _CONTRADICTION_TEMPLATES[contradiction_type]
        logger.info(f"Contradiction detected: {template['label']} (severity: {template['severity']})")
        return {
            "has_contradiction": True,
            "contradiction_type": contradiction_type,
            "label": template["label"],
            "explanation": template["explanation"],
            "severity": template["severity"],
            "debug": {
                "physio_stress": physio_stress_count,
                "physio_calm": physio_calm_count,
                "behav_stress": behav_stress_count,
                "behav_calm": behav_calm_count,
                "behav_fatigue": behav_fatigue_count,
            },
        }

    logger.info("No cross-modal contradiction detected.")
    return {
        "has_contradiction": False,
        "contradiction_type": None,
        "label": "Signals Aligned",
        "explanation": "Physiological and behavioral signals are telling a consistent story.",
        "severity": "none",
    }
