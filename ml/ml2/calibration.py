"""
calibration.py
--------------
User Baseline Calibration — records the user's personal baseline vitals
during a calm state, then uses them for personalized triage.
"""

import os
import sys
import json
import time
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("ml2.calibration")

# Path to store user profile
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "user_profile.json")


class UserProfile:
    """
    Stores and manages a user's personal baseline readings.
    Used to personalize triage thresholds and wellness scoring.
    """

    def __init__(self):
        self.is_calibrated = False
        self.calibrated_at = None
        self.baseline = {
            "hr": 72.0,
            "bp_sys": 120.0,
            "bp_dia": 80.0,
            "respiration": 15.0,
            "blink_rate": 17.0,
            "gaze_stability": 0.75,
            "eye_closure": 0.08,
        }
        self.calibration_readings = []

    def add_calibration_reading(self, vitals: dict):
        """
        Add a single reading during calibration phase.
        Call this repeatedly during the 60-second calibration window.
        """
        physio = vitals.get("physio", {})
        behavioral = vitals.get("behavioral", {})

        reading = {
            "timestamp": time.time(),
            "hr": physio.get("hr"),
            "bp_sys": physio.get("bp_sys"),
            "bp_dia": physio.get("bp_dia"),
            "respiration": physio.get("respiration"),
            "blink_rate": behavioral.get("blink_rate"),
            "gaze_stability": behavioral.get("gaze_stability"),
            "eye_closure": behavioral.get("eye_closure"),
        }
        self.calibration_readings.append(reading)

    def finalize_calibration(self) -> dict:
        """
        Computes the baseline from all calibration readings.
        Call this after the calibration window ends.

        Returns:
            dict: The computed baseline values.
        """
        if len(self.calibration_readings) < 3:
            logger.warning("Not enough calibration readings (need >= 3). Using defaults.")
            return self.baseline

        # Compute median for each signal (more robust than mean to outliers)
        for key in self.baseline.keys():
            values = [r[key] for r in self.calibration_readings if r.get(key) is not None]
            if values:
                values.sort()
                mid = len(values) // 2
                if len(values) % 2 == 0:
                    self.baseline[key] = round((values[mid - 1] + values[mid]) / 2.0, 2)
                else:
                    self.baseline[key] = round(values[mid], 2)

        self.is_calibrated = True
        self.calibrated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        logger.info(f"Calibration complete. Baseline: {self.baseline}")
        self.save()
        return self.baseline

    def get_baseline(self) -> dict:
        """Returns the current baseline (calibrated or defaults)."""
        return self.baseline.copy()

    def get_deviation_report(self, vitals: dict) -> dict:
        """
        Compares current vitals against the personal baseline.

        Returns:
            dict with per-signal deviation info for the Judge prompt.
        """
        physio = vitals.get("physio", {})
        behavioral = vitals.get("behavioral", {})

        current = {
            "hr": physio.get("hr"),
            "bp_sys": physio.get("bp_sys"),
            "bp_dia": physio.get("bp_dia"),
            "respiration": physio.get("respiration"),
            "blink_rate": behavioral.get("blink_rate"),
            "gaze_stability": behavioral.get("gaze_stability"),
            "eye_closure": behavioral.get("eye_closure"),
        }

        deviations = {}
        labels = {
            "hr": "Heart Rate",
            "bp_sys": "Systolic BP",
            "bp_dia": "Diastolic BP",
            "respiration": "Respiration Rate",
            "blink_rate": "Blink Rate",
            "gaze_stability": "Gaze Stability",
            "eye_closure": "Eye Closure",
        }

        for key, label in labels.items():
            baseline_val = self.baseline.get(key)
            current_val = current.get(key)

            if baseline_val is None or current_val is None:
                continue

            if baseline_val == 0:
                pct_change = 0
            else:
                pct_change = round(((current_val - baseline_val) / abs(baseline_val)) * 100, 1)

            deviation = {
                "label": label,
                "baseline": baseline_val,
                "current": round(current_val, 2),
                "change_pct": pct_change,
                "status": "normal",
            }

            # Flag significant deviations (>20%)
            if abs(pct_change) > 20:
                deviation["status"] = "elevated" if pct_change > 0 else "reduced"

            deviations[key] = deviation

        return {
            "is_calibrated": self.is_calibrated,
            "deviations": deviations,
        }

    def get_baseline_context_for_prompt(self, vitals: dict) -> str:
        """
        Generates a natural language string for injecting into agent prompts.
        Compares current readings against the user's personal baseline.
        """
        if not self.is_calibrated:
            return ""

        report = self.get_deviation_report(vitals)
        lines = ["User's Personal Baseline (measured during calm state):"]

        for key, dev in report.get("deviations", {}).items():
            status_icon = ""
            if dev["status"] == "elevated":
                status_icon = "(ABOVE baseline)"
            elif dev["status"] == "reduced":
                status_icon = "(BELOW baseline)"

            lines.append(
                f"  - {dev['label']}: baseline={dev['baseline']}, "
                f"current={dev['current']} ({dev['change_pct']:+.0f}%) {status_icon}"
            )

        return "\n".join(lines)

    def save(self):
        """Persist profile to disk."""
        data = {
            "is_calibrated": self.is_calibrated,
            "calibrated_at": self.calibrated_at,
            "baseline": self.baseline,
        }
        try:
            with open(PROFILE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Profile saved to {PROFILE_PATH}")
        except Exception as e:
            logger.warning(f"Failed to save profile: {e}")

    def load(self):
        """Load profile from disk if it exists."""
        if os.path.exists(PROFILE_PATH):
            try:
                with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.is_calibrated = data.get("is_calibrated", False)
                self.calibrated_at = data.get("calibrated_at")
                self.baseline = data.get("baseline", self.baseline)
                logger.info(f"Profile loaded (calibrated: {self.is_calibrated})")
            except Exception as e:
                logger.warning(f"Failed to load profile: {e}")

    def to_dict(self) -> dict:
        """Serialize to dict for API responses."""
        return {
            "is_calibrated": self.is_calibrated,
            "calibrated_at": self.calibrated_at,
            "baseline": self.baseline,
        }


# Global singleton
_PROFILE = UserProfile()
_PROFILE.load()  # Auto-load on import


def get_profile() -> UserProfile:
    """Returns the global user profile singleton."""
    return _PROFILE


def reset_profile():
    """Resets the user profile to defaults."""
    global _PROFILE
    _PROFILE = UserProfile()
    # Remove saved file
    if os.path.exists(PROFILE_PATH):
        os.remove(PROFILE_PATH)
