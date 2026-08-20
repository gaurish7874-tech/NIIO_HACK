"""
report.py
---------
Session Report Generator — creates a structured wellness report
summarizing an entire monitoring session.
"""

import time
import logging

logger = logging.getLogger("ml2.report")


def generate_session_report(timeline, latest_triage=None, latest_score=None,
                            latest_emotional=None, latest_contradiction=None) -> dict:
    """
    Generates a comprehensive session report from timeline data.

    Parameters:
        timeline: WellnessTimeline instance with session history.
        latest_triage: Most recent TriageResult.
        latest_score: Most recent wellness score dict.
        latest_emotional: Most recent emotional map dict.
        latest_contradiction: Most recent contradiction dict.

    Returns:
        dict: Structured session report ready for frontend rendering.
    """
    stats = timeline.get_session_stats()
    trends = timeline.get_trend_summary()

    if not stats:
        return {
            "status": "no_data",
            "message": "No monitoring data available for report generation.",
        }

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "complete",

        # Session overview
        "session": {
            "duration_minutes": stats.get("duration_minutes", 0),
            "total_readings": stats.get("total_readings", 0),
        },

        # Signal statistics
        "vitals_summary": {},

        # Verdict distribution
        "verdict_distribution": stats.get("verdict_counts", {}),

        # Trend analysis
        "trends": {},

        # Current state
        "current_state": {},

        # Key events / alerts
        "alerts": trends.get("alerts", []),

        # Recommendations
        "recommendations": [],
    }

    # Build vitals summary
    signal_labels = {
        "hr": {"label": "Heart Rate", "unit": "BPM", "normal": "60-100"},
        "hrv": {"label": "HRV (RMSSD)", "unit": "ms", "normal": "20-100"},
        "respiration": {"label": "Respiration", "unit": "RPM", "normal": "12-20"},
        "blink_rate": {"label": "Blink Rate", "unit": "/min", "normal": "15-20"},
        "gaze_stability": {"label": "Gaze Stability", "unit": "", "normal": "0.6-1.0"},
        "eye_closure": {"label": "Eye Closure", "unit": "", "normal": "0.0-0.2"},
        "wellness_score": {"label": "Wellness Score", "unit": "/100", "normal": "70-100"},
    }

    for key, meta in signal_labels.items():
        sig_stats = stats.get("signals", {}).get(key)
        if sig_stats:
            report["vitals_summary"][key] = {
                "label": meta["label"],
                "unit": meta["unit"],
                "normal_range": meta["normal"],
                "min": sig_stats["min"],
                "max": sig_stats["max"],
                "avg": sig_stats["avg"],
                "latest": sig_stats["latest"],
            }

    # Build trend summary
    for key, trend_data in trends.get("trends", {}).items():
        if key in signal_labels:
            report["trends"][key] = {
                "label": signal_labels[key]["label"],
                "direction": trend_data.get("direction", "unknown"),
                "change_pct": trend_data.get("change_pct", 0),
            }

    # Current state snapshot
    if latest_triage:
        report["current_state"]["verdict"] = (
            latest_triage.verdict.value if hasattr(latest_triage, 'verdict') and hasattr(latest_triage.verdict, 'value')
            else str(latest_triage.verdict) if hasattr(latest_triage, 'verdict')
            else "unknown"
        )
        report["current_state"]["confidence"] = (
            latest_triage.confidence if hasattr(latest_triage, 'confidence') else 0
        )
        report["current_state"]["summary"] = (
            latest_triage.summary if hasattr(latest_triage, 'summary') else ""
        )

    if latest_score:
        report["current_state"]["wellness_score"] = latest_score.get("score", 0)
        report["current_state"]["wellness_category"] = latest_score.get("category", "unknown")

    if latest_emotional:
        report["current_state"]["emotional_quadrant"] = latest_emotional.get("quadrant_label", "")
        report["current_state"]["arousal"] = latest_emotional.get("arousal", 0)
        report["current_state"]["valence"] = latest_emotional.get("valence", 0)

    if latest_contradiction and latest_contradiction.get("has_contradiction"):
        report["current_state"]["contradiction"] = latest_contradiction.get("label", "")

    # Generate recommendations based on verdict distribution
    verdict_counts = stats.get("verdict_counts", {})
    total_verdicts = sum(verdict_counts.values())

    if total_verdicts > 0:
        stress_pct = (verdict_counts.get("mild_stress", 0) + verdict_counts.get("high_stress", 0) +
                      verdict_counts.get("anxiety", 0)) / total_verdicts * 100
        fatigue_pct = (verdict_counts.get("fatigue", 0) + verdict_counts.get("drowsiness", 0)) / total_verdicts * 100
        normal_pct = verdict_counts.get("normal", 0) / total_verdicts * 100

        if stress_pct > 50:
            report["recommendations"].append({
                "priority": "high",
                "category": "stress",
                "message": (
                    f"Stress signals were detected in {stress_pct:.0f}% of readings during this session. "
                    "Consider incorporating regular breaks and breathing exercises into your routine."
                ),
            })
        if fatigue_pct > 30:
            report["recommendations"].append({
                "priority": "high",
                "category": "fatigue",
                "message": (
                    f"Fatigue or drowsiness was detected in {fatigue_pct:.0f}% of readings. "
                    "Ensure you're getting adequate sleep (7-9 hours) and taking regular screen breaks."
                ),
            })
        if normal_pct > 70:
            report["recommendations"].append({
                "priority": "low",
                "category": "positive",
                "message": (
                    f"Great session! You were in a healthy state for {normal_pct:.0f}% of the monitoring period. "
                    "Keep up your current wellness habits."
                ),
            })

        # Trend-based recommendations
        score_trend = trends.get("score_trend", "stable")
        if score_trend == "declining":
            report["recommendations"].append({
                "priority": "medium",
                "category": "trend",
                "message": "Your overall wellness score has been declining during this session. Consider taking a break.",
            })

    logger.info(f"Session report generated: {stats.get('total_readings', 0)} readings, "
                f"{stats.get('duration_minutes', 0)} min")
    return report


def format_report_text(report: dict) -> str:
    """Formats a session report as readable plain text."""
    if report.get("status") == "no_data":
        return report.get("message", "No data available.")

    lines = []
    lines.append("=" * 60)
    lines.append("  WELLNESS MONITORING SESSION REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {report.get('generated_at', 'N/A')}")
    lines.append(f"Duration: {report['session']['duration_minutes']} minutes")
    lines.append(f"Total Readings: {report['session']['total_readings']}")

    # Vitals Summary
    lines.append("\n--- Vitals Summary ---")
    for key, data in report.get("vitals_summary", {}).items():
        lines.append(
            f"  {data['label']}: avg={data['avg']}{data['unit']}, "
            f"range=[{data['min']}-{data['max']}], normal={data['normal_range']}"
        )

    # Verdict Distribution
    lines.append("\n--- Verdict Distribution ---")
    for verdict, count in report.get("verdict_distribution", {}).items():
        lines.append(f"  {verdict}: {count}")

    # Current State
    state = report.get("current_state", {})
    if state:
        lines.append("\n--- Current State ---")
        if "verdict" in state:
            lines.append(f"  Verdict: {state['verdict']}")
        if "wellness_score" in state:
            lines.append(f"  Wellness Score: {state['wellness_score']}/100 ({state.get('wellness_category', '')})")
        if "emotional_quadrant" in state:
            lines.append(f"  Emotional State: {state['emotional_quadrant']}")

    # Alerts
    if report.get("alerts"):
        lines.append("\n--- Alerts ---")
        for alert in report["alerts"]:
            lines.append(f"  [!] {alert}")

    # Recommendations
    if report.get("recommendations"):
        lines.append("\n--- Recommendations ---")
        for rec in report["recommendations"]:
            priority = rec["priority"].upper()
            lines.append(f"  [{priority}] {rec['message']}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
