"""
timeline.py
-----------
Temporal Trend Tracking — stores vitals history and detects trends.
Transforms ML-2 from a point-in-time snapshot to a continuous monitor.
"""

import time
import logging
from collections import deque

logger = logging.getLogger("ml2.timeline")


class WellnessTimeline:
    """
    Stores recent triage snapshots and detects temporal trends.
    Thread-safe for use in a FastAPI background context.
    """

    def __init__(self, max_history: int = 60):
        """
        Args:
            max_history: Maximum number of snapshots to retain.
                         At 1 snapshot per 5-10 seconds, 60 = ~5-10 minutes of history.
        """
        self.max_history = max_history
        self.history = deque(maxlen=max_history)

    def add_snapshot(self, vitals: dict, verdict: str, wellness_score: int):
        """Record a new triage snapshot."""
        self.history.append({
            "timestamp": time.time(),
            "hr": vitals.get("physio", {}).get("hr"),
            "bp_sys": vitals.get("physio", {}).get("bp_sys"),
            "bp_dia": vitals.get("physio", {}).get("bp_dia"),
            "respiration": vitals.get("physio", {}).get("respiration"),
            "blink_rate": vitals.get("behavioral", {}).get("blink_rate"),
            "gaze_stability": vitals.get("behavioral", {}).get("gaze_stability"),
            "eye_closure": vitals.get("behavioral", {}).get("eye_closure"),
            "verdict": verdict,
            "wellness_score": wellness_score,
        })

    def get_trend_summary(self) -> dict:
        """
        Analyzes the history and returns trend information.

        Returns:
            dict with:
                has_enough_data (bool)
                duration_minutes (float)
                snapshot_count (int)
                trends (dict): per-signal trend direction
                trend_narrative (str): natural language summary for the Judge
                verdict_history (list): recent verdicts
                score_trend (str): "improving" | "stable" | "declining"
                alerts (list): significant changes detected
        """
        if len(self.history) < 3:
            return {
                "has_enough_data": False,
                "duration_minutes": 0,
                "snapshot_count": len(self.history),
                "trends": {},
                "trend_narrative": "Not enough data for trend analysis yet.",
                "verdict_history": [],
                "score_trend": "unknown",
                "alerts": [],
            }

        snapshots = list(self.history)
        duration_sec = snapshots[-1]["timestamp"] - snapshots[0]["timestamp"]
        duration_min = round(duration_sec / 60.0, 1)

        # Split into early (first third) and recent (last third)
        n = len(snapshots)
        third = max(1, n // 3)
        early = snapshots[:third]
        recent = snapshots[-third:]

        trends = {}
        alerts = []

        signal_keys = ["hr", "bp_sys", "bp_dia", "respiration", "blink_rate", "gaze_stability",
                       "eye_closure", "wellness_score"]
        signal_labels = {
            "hr": "Heart Rate",
            "bp_sys": "Systolic BP",
            "bp_dia": "Diastolic BP",
            "respiration": "Respiration Rate",
            "blink_rate": "Blink Rate",
            "gaze_stability": "Gaze Stability",
            "eye_closure": "Eye Closure",
            "wellness_score": "Wellness Score",
        }

        for key in signal_keys:
            early_vals = [s[key] for s in early if s.get(key) is not None]
            recent_vals = [s[key] for s in recent if s.get(key) is not None]

            if not early_vals or not recent_vals:
                trends[key] = {"direction": "unknown", "change_pct": 0}
                continue

            early_avg = sum(early_vals) / len(early_vals)
            recent_avg = sum(recent_vals) / len(recent_vals)

            if early_avg == 0:
                change_pct = 0
            else:
                change_pct = round(((recent_avg - early_avg) / abs(early_avg)) * 100, 1)

            # Determine direction
            if abs(change_pct) < 5:
                direction = "stable"
            elif change_pct > 0:
                direction = "increasing"
            else:
                direction = "decreasing"

            trends[key] = {
                "direction": direction,
                "change_pct": change_pct,
                "early_avg": round(early_avg, 1),
                "recent_avg": round(recent_avg, 1),
            }

            # Generate alerts for significant changes
            label = signal_labels.get(key, key)
            if abs(change_pct) >= 15:
                alerts.append(
                    f"{label} has {'increased' if change_pct > 0 else 'decreased'} "
                    f"by {abs(change_pct):.0f}% over the last {duration_min} minutes "
                    f"(from {round(early_avg, 1)} to {round(recent_avg, 1)})"
                )

        # Wellness score trend
        ws_trend = trends.get("wellness_score", {})
        if ws_trend.get("direction") == "increasing":
            score_trend = "improving"
        elif ws_trend.get("direction") == "decreasing":
            score_trend = "declining"
        else:
            score_trend = "stable"

        # Verdict history (last 10)
        verdict_history = [s["verdict"] for s in snapshots[-10:]]

        # Build natural language narrative for the Judge
        narrative_parts = []
        if duration_min >= 1:
            narrative_parts.append(f"Monitoring for {duration_min} minutes ({len(snapshots)} readings).")

        for key in ["hr", "bp_sys", "wellness_score"]:
            t = trends.get(key, {})
            label = signal_labels.get(key, key)
            if t.get("direction") not in ("stable", "unknown") and abs(t.get("change_pct", 0)) >= 10:
                narrative_parts.append(
                    f"{label} is {t['direction']} ({t['change_pct']:+.0f}%, "
                    f"from {t.get('early_avg', '?')} to {t.get('recent_avg', '?')})."
                )

        if score_trend == "declining":
            narrative_parts.append("Overall wellness trend is declining.")
        elif score_trend == "improving":
            narrative_parts.append("Overall wellness trend is improving.")

        # Sustained verdict detection
        if len(verdict_history) >= 5:
            recent_5 = verdict_history[-5:]
            if all(v == recent_5[0] for v in recent_5):
                narrative_parts.append(
                    f"Verdict has been consistently '{recent_5[0]}' for the last {len(recent_5)} readings."
                )

        trend_narrative = " ".join(narrative_parts) if narrative_parts else "All signals are stable."

        logger.info(f"Trend analysis: {score_trend}, {len(alerts)} alerts, {duration_min} min history")

        return {
            "has_enough_data": True,
            "duration_minutes": duration_min,
            "snapshot_count": len(snapshots),
            "trends": trends,
            "trend_narrative": trend_narrative,
            "verdict_history": verdict_history,
            "score_trend": score_trend,
            "alerts": alerts,
        }

    def get_session_stats(self) -> dict:
        """Returns aggregate statistics for the entire session (used by report generator)."""
        if not self.history:
            return {}

        snapshots = list(self.history)
        duration_sec = snapshots[-1]["timestamp"] - snapshots[0]["timestamp"]

        stats = {
            "duration_minutes": round(duration_sec / 60.0, 1),
            "total_readings": len(snapshots),
            "signals": {},
            "verdict_counts": {},
            "score_stats": {},
        }

        # Signal statistics
        for key in ["hr", "bp_sys", "bp_dia", "respiration", "blink_rate", "gaze_stability",
                     "eye_closure", "wellness_score"]:
            vals = [s[key] for s in snapshots if s.get(key) is not None]
            if vals:
                stats["signals"][key] = {
                    "min": round(min(vals), 1),
                    "max": round(max(vals), 1),
                    "avg": round(sum(vals) / len(vals), 1),
                    "latest": round(vals[-1], 1),
                }

        # Verdict distribution
        for s in snapshots:
            v = s["verdict"]
            stats["verdict_counts"][v] = stats["verdict_counts"].get(v, 0) + 1

        return stats


# Global singleton timeline instance
_TIMELINE = WellnessTimeline()


def get_timeline() -> WellnessTimeline:
    """Returns the global timeline singleton."""
    return _TIMELINE


def reset_timeline():
    """Resets the global timeline (e.g., for new session)."""
    global _TIMELINE
    _TIMELINE = WellnessTimeline()
