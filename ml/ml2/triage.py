"""
triage.py
---------
Module responsibility (ML-2):
Agentic Debate Triage Engine — uses Groq-hosted LLMs to run a multi-agent
debate (Physio Agent vs Behavioral Agent → Judge) for explainable triage.
"""

import os
import sys
import json
import logging

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from openai import OpenAI

from ml.ml2.schemas import (
    AgentArgument,
    JudgeDecision,
    TriageResult,
    TriageVerdict,
)
from ml.ml2.prompts import (
    PHYSIO_AGENT_SYSTEM_PROMPT,
    BEHAVIORAL_AGENT_SYSTEM_PROMPT,
    JUDGE_AGENT_SYSTEM_PROMPT,
    JUDGE_AGENT_FOLLOWUP_PROMPT,
)

# Load environment variables
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TEMPERATURE = 0.3
GROQ_MAX_TOKENS = 512

logger = logging.getLogger("ml2.triage")


def _get_groq_client() -> OpenAI:
    """Returns an OpenAI-compatible client pointed at Groq."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found. Set it in .env or as an environment variable. "
            "Get a free key at https://console.groq.com"
        )
    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    """
    Low-level Groq API call.
    Returns the raw text content from the LLM response.
    """
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content.strip()


def _parse_json_response(raw: str) -> dict:
    """
    Extracts JSON from LLM response, handling markdown code fences.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _run_physio_agent(physio_data: dict) -> AgentArgument:
    """
    Runs the Physiological Agent on the physio signals.
    Returns a structured AgentArgument.
    """
    user_prompt = f"""Analyze the following physiological vital signs and provide your assessment:

{json.dumps(physio_data, indent=2)}

Remember: respond ONLY with the JSON format specified in your instructions."""

    raw_response = _call_groq(PHYSIO_AGENT_SYSTEM_PROMPT, user_prompt)
    logger.debug(f"Physio Agent raw response: {raw_response}")

    try:
        parsed = _parse_json_response(raw_response)
        return AgentArgument(**parsed)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse Physio Agent response: {e}. Using fallback.")
        return AgentArgument(
            agent_name="Physio Agent",
            assessment="unable to assess (parse error)",
            reasoning=f"The agent produced a response that could not be parsed: {raw_response[:200]}",
            confidence=0.1,
            key_signals=[],
        )


def _run_behavioral_agent(behavioral_data: dict) -> AgentArgument:
    """
    Runs the Behavioral Agent on the behavioral signals.
    Returns a structured AgentArgument.
    """
    user_prompt = f"""Analyze the following behavioral signals and provide your assessment:

{json.dumps(behavioral_data, indent=2)}

Remember: respond ONLY with the JSON format specified in your instructions."""

    raw_response = _call_groq(BEHAVIORAL_AGENT_SYSTEM_PROMPT, user_prompt)
    logger.debug(f"Behavioral Agent raw response: {raw_response}")

    try:
        parsed = _parse_json_response(raw_response)
        return AgentArgument(**parsed)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse Behavioral Agent response: {e}. Using fallback.")
        return AgentArgument(
            agent_name="Behavioral Agent",
            assessment="unable to assess (parse error)",
            reasoning=f"The agent produced a response that could not be parsed: {raw_response[:200]}",
            confidence=0.1,
            key_signals=[],
        )


def _run_judge_agent(
    physio_arg: AgentArgument,
    behavioral_arg: AgentArgument,
    user_answer: str = "",
) -> JudgeDecision:
    """
    Runs the Judge Agent to evaluate both agent arguments and produce a verdict.
    If user_answer is provided, uses the followup prompt for re-evaluation.
    """
    # Choose prompt based on whether this is initial or followup
    if user_answer:
        system_prompt = JUDGE_AGENT_FOLLOWUP_PROMPT
    else:
        system_prompt = JUDGE_AGENT_SYSTEM_PROMPT

    # Build user prompt
    user_prompt = f"""## Physio Agent's Argument:
{json.dumps(physio_arg.model_dump(), indent=2)}

## Behavioral Agent's Argument:
{json.dumps(behavioral_arg.model_dump(), indent=2)}"""

    if user_answer:
        user_prompt += f"""

## User's Answer to Your Previous Question:
"{user_answer}"

Incorporate this answer into your final verdict. Do NOT ask another question."""

    user_prompt += "\n\nRemember: respond ONLY with the JSON format specified in your instructions."

    raw_response = _call_groq(system_prompt, user_prompt)
    logger.debug(f"Judge Agent raw response: {raw_response}")

    try:
        parsed = _parse_json_response(raw_response)
        return JudgeDecision(**parsed)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse Judge Agent response: {e}. Using fallback.")
        return JudgeDecision(
            verdict=TriageVerdict.NEEDS_ATTENTION,
            confidence=0.2,
            summary="Unable to produce a reliable assessment. Please try again.",
            needs_clarification=False,
            clarifying_question=None,
            physio_weight=0.5,
            behavioral_weight=0.5,
        )


def predict_triage_score(vitals: dict, user_answer: str = "") -> TriageResult:
    """
    Main entry point for ML-2 Triage.

    Orchestrates the full enriched pipeline:
    1. Compute wellness score
    2. Detect cross-modal contradictions
    3. Compute arousal-valence emotional map
    4. Get baseline deviation (if calibrated)
    5. Run Physio Agent + Behavioral Agent (with enriched context)
    6. Run Judge Agent (with contradiction + trend context)
    7. Generate RAG guidance
    8. Record to timeline

    Parameters:
        vitals (dict): The multimodal JSON from ML-1's analyze_multimodal_wellness().
                       Must contain "physio" and "behavioral" keys.
        user_answer (str): Optional answer to a previously asked clarifying question.

    Returns:
        TriageResult: Complete enriched triage output.
    """
    logger.info("Starting ML-2 Enriched Triage pipeline...")

    # Extract sub-dicts
    physio_data = vitals.get("physio", {})
    behavioral_data = vitals.get("behavioral", {})

    # --- Step 1: Compute Wellness Score ---
    wellness_score_data = None
    try:
        from ml.ml2.wellness_score import compute_wellness_score
        from ml.ml2.calibration import get_profile
        profile = get_profile()
        baseline = profile.get_baseline() if profile.is_calibrated else None
        wellness_score_data = compute_wellness_score(vitals, baseline=baseline)
        logger.info(f"Wellness Score: {wellness_score_data['score']}/100 ({wellness_score_data['category']})")
    except Exception as e:
        logger.warning(f"Wellness score computation failed (non-fatal): {e}")

    # --- Step 2: Detect Cross-Modal Contradictions ---
    contradiction_data = None
    try:
        from ml.ml2.contradiction import detect_contradiction
        contradiction_data = detect_contradiction(vitals)
        if contradiction_data.get("has_contradiction"):
            logger.info(f"Contradiction detected: {contradiction_data['label']}")
    except Exception as e:
        logger.warning(f"Contradiction detection failed (non-fatal): {e}")

    # --- Step 3: Compute Arousal-Valence Emotional Map ---
    emotional_map_data = None
    try:
        from ml.ml2.emotional_map import compute_arousal_valence
        from ml.ml2.calibration import get_profile
        profile = get_profile()
        baseline = profile.get_baseline() if profile.is_calibrated else None
        emotional_map_data = compute_arousal_valence(vitals, baseline=baseline)
        logger.info(f"Emotional map: {emotional_map_data['quadrant_label']}")
    except Exception as e:
        logger.warning(f"Emotional mapping failed (non-fatal): {e}")

    # --- Step 4: Get Baseline Deviation ---
    baseline_deviation_data = None
    baseline_context = ""
    try:
        from ml.ml2.calibration import get_profile
        profile = get_profile()
        if profile.is_calibrated:
            baseline_deviation_data = profile.get_deviation_report(vitals)
            baseline_context = profile.get_baseline_context_for_prompt(vitals)
            logger.info("Baseline deviation computed.")
    except Exception as e:
        logger.warning(f"Baseline deviation failed (non-fatal): {e}")

    # --- Step 5: Get Temporal Trend Summary ---
    trend_summary_data = None
    trend_context = ""
    try:
        from ml.ml2.timeline import get_timeline
        timeline = get_timeline()
        trend_summary_data = timeline.get_trend_summary()
        if trend_summary_data.get("has_enough_data"):
            trend_context = trend_summary_data.get("trend_narrative", "")
            logger.info(f"Trend: {trend_summary_data['score_trend']}")
    except Exception as e:
        logger.warning(f"Trend analysis failed (non-fatal): {e}")

    # --- Step 6: Run both specialist agents ---
    logger.info("Running Physio Agent...")
    physio_arg = _run_physio_agent(physio_data)
    logger.info(f"Physio Agent assessment: {physio_arg.assessment}")

    logger.info("Running Behavioral Agent...")
    behavioral_arg = _run_behavioral_agent(behavioral_data)
    logger.info(f"Behavioral Agent assessment: {behavioral_arg.assessment}")

    # --- Step 7: Run Judge Agent (with enriched context) ---
    # Inject contradiction, trend, and baseline context into the Judge call
    enriched_context = ""
    if contradiction_data and contradiction_data.get("has_contradiction"):
        enriched_context += (
            f"\n## Cross-Modal Contradiction Detected\n"
            f"Type: {contradiction_data['label']}\n"
            f"Analysis: {contradiction_data['explanation']}\n"
            f"Severity: {contradiction_data['severity']}\n"
            f"Consider this contradiction in your verdict.\n"
        )
    if trend_context:
        enriched_context += f"\n## Temporal Trend Context\n{trend_context}\n"
    if baseline_context:
        enriched_context += f"\n## {baseline_context}\n"
    if wellness_score_data:
        enriched_context += (
            f"\n## Wellness Score: {wellness_score_data['score']}/100 "
            f"({wellness_score_data['category']})\n"
        )

    # Pass enriched context as additional user_answer context
    combined_context = user_answer
    if enriched_context and not user_answer:
        combined_context = enriched_context
    elif enriched_context and user_answer:
        combined_context = user_answer + "\n\n" + enriched_context

    logger.info("Running Judge Agent...")
    judge_decision = _run_judge_agent(physio_arg, behavioral_arg, combined_context)
    logger.info(f"Judge verdict: {judge_decision.verdict} (confidence: {judge_decision.confidence})")

    # --- Step 8: Build triage result ---
    result = TriageResult(
        verdict=judge_decision.verdict,
        confidence=judge_decision.confidence,
        summary=judge_decision.summary,
        debate_log=[physio_arg, behavioral_arg],
        judge_decision=judge_decision,
        needs_clarification=judge_decision.needs_clarification,
        clarifying_question=judge_decision.clarifying_question,
        wellness_score=wellness_score_data,
        contradiction=contradiction_data,
        emotional_map=emotional_map_data,
        trend_summary=trend_summary_data,
        baseline_deviation=baseline_deviation_data,
    )

    # --- Step 9: Generate RAG guidance (if not needing clarification) ---
    if not judge_decision.needs_clarification:
        try:
            from ml.ml2.rag_layer import generate_guidance
            guidance_result = generate_guidance(judge_decision.verdict, judge_decision.summary)
            result.guidance = guidance_result.advice
            result.guidance_sources = guidance_result.sources
            logger.info("RAG guidance generated successfully.")
        except Exception as e:
            logger.warning(f"RAG guidance generation failed (non-fatal): {e}")
            result.guidance = None
            result.guidance_sources = []

    # --- Step 10: Record to timeline ---
    try:
        from ml.ml2.timeline import get_timeline
        timeline = get_timeline()
        ws = wellness_score_data.get("score", 50) if wellness_score_data else 50
        timeline.add_snapshot(vitals, judge_decision.verdict.value, ws)
        logger.info("Snapshot recorded to timeline.")
    except Exception as e:
        logger.warning(f"Timeline recording failed (non-fatal): {e}")

    logger.info("ML-2 Enriched Triage pipeline complete.")
    return result


# -----------------------------------------------------------------------
# Standalone Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    print("=" * 70)
    print("  ML-2 ENRICHED TRIAGE ENGINE -- Integration Test")
    print("=" * 70)

    # Test Case 1: Stressed vitals
    test_vitals_stressed = {
        "timestamp": "2026-08-19T18:00:00Z",
        "physio": {
            "hr": 105.0,
            "bp_sys": 145.0,
            "bp_dia": 90.0,
            "respiration": 22.0,
            "confidence": 0.85,
        },
        "behavioral": {
            "blink_rate": 28.0,
            "eye_closure": 0.12,
            "gaze_stability": 0.35,
            "head_pose": {"pitch": -2.0, "yaw": 8.0, "roll": 1.5},
            "confidence": 0.82,
        },
    }

    print("\n--- Test Case: STRESSED vitals ---")
    result = predict_triage_score(test_vitals_stressed)
    print(f"\nVerdict: {result.verdict}")
    print(f"Confidence: {result.confidence}")
    print(f"Summary: {result.summary}")

    # Wellness Score
    if result.wellness_score:
        ws = result.wellness_score
        print(f"\nWellness Score: {ws['score']}/100 ({ws['category']}) {ws['color']}")
        print(f"  Breakdown: {ws['breakdown']}")

    # Emotional Map
    if result.emotional_map:
        em = result.emotional_map
        print(f"\nEmotional State: {em['quadrant_label']}")
        print(f"  Arousal: {em['arousal']}, Valence: {em['valence']}")
        print(f"  {em['description']}")

    # Contradiction
    if result.contradiction:
        ct = result.contradiction
        if ct.get("has_contradiction"):
            print(f"\nContradiction: {ct['label']} (severity: {ct['severity']})")
            print(f"  {ct['explanation']}")
        else:
            print(f"\nContradiction: None (signals aligned)")

    # Debate Log
    print("\n--- Debate Log ---")
    for arg in result.debate_log:
        print(f"  [{arg.agent_name}] {arg.assessment} (conf: {arg.confidence})")
        for sig in arg.key_signals:
            print(f"    - {sig}")

    # RAG Guidance
    if result.guidance:
        safe_guidance = result.guidance.encode("ascii", errors="replace").decode("ascii")
        print(f"\n--- Guidance ---\n{safe_guidance}")
        print(f"Sources: {result.guidance_sources}")

    # Test Case 2: Contradiction scenario (physio stressed, behavioral calm)
    print("\n" + "=" * 70)
    print("--- Test Case: CONTRADICTION (physio stressed, behavioral calm) ---")
    test_vitals_contradiction = {
        "timestamp": "2026-08-19T18:05:00Z",
        "physio": {
            "hr": 102.0,
            "bp_sys": 140.0,
            "bp_dia": 88.0,
            "respiration": 21.0,
            "confidence": 0.80,
        },
        "behavioral": {
            "blink_rate": 16.0,
            "eye_closure": 0.08,
            "gaze_stability": 0.82,
            "head_pose": {"pitch": -1.0, "yaw": 2.0, "roll": 0.5},
            "confidence": 0.90,
        },
    }
    result2 = predict_triage_score(test_vitals_contradiction)
    print(f"Verdict: {result2.verdict}")
    if result2.contradiction and result2.contradiction.get("has_contradiction"):
        print(f"Contradiction: {result2.contradiction['label']}")
        print(f"  {result2.contradiction['explanation'][:150]}...")
    if result2.wellness_score:
        print(f"Wellness Score: {result2.wellness_score['score']}/100")
    if result2.emotional_map:
        print(f"Emotional State: {result2.emotional_map['quadrant_label']}")

    print("\n[SUCCESS] ML-2 Enriched Triage pipeline test complete!")

