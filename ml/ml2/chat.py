"""
chat.py
-------
Conversational Wellness Chat — multi-turn conversation about the user's
current wellness state, grounded in triage results and RAG knowledge.
"""

import os
import sys
import json
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logger = logging.getLogger("ml2.chat")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

WELLNESS_CHAT_SYSTEM_PROMPT = """You are a friendly, knowledgeable Wellness Assistant in a contactless health monitoring system.

You have access to the user's current physiological and behavioral readings, the triage verdict from our AI agents, and wellness guidance from our knowledge base.

## Rules:
1. Be warm, empathetic, and conversational — not clinical.
2. Answer questions about the user's readings in plain language.
3. If asked about medical diagnoses, always clarify you are NOT a doctor and recommend consulting a healthcare professional.
4. Reference the actual data when answering (e.g., "Your heart rate is currently 95 BPM, which is slightly elevated").
5. Keep responses concise — 2-4 sentences unless asked for detail.
6. If the user asks about something outside your context, honestly say you don't have that information.
7. You can suggest breathing exercises, breaks, and wellness tips from the knowledge base.
8. Never make up readings or data not in the context provided."""


class WellnessChat:
    """
    Multi-turn conversational wellness assistant.
    Maintains chat history and injects current triage context.
    """

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.chat_history = []
        self._context = ""

    def set_context(self, vitals: dict, triage_result=None, wellness_score: dict = None,
                    emotional_map: dict = None, contradiction: dict = None,
                    trend_summary: dict = None):
        """
        Updates the wellness context injected into every chat message.
        Call this after each triage cycle.
        """
        context_parts = ["## Current User Wellness Data"]

        # Vitals
        physio = vitals.get("physio", {})
        behavioral = vitals.get("behavioral", {})
        context_parts.append(f"""
### Physiological Readings
- Heart Rate: {physio.get('hr', 'N/A')} BPM
- HRV (RMSSD): {physio.get('hrv', 'N/A')} ms
- Respiration Rate: {physio.get('respiration', 'N/A')} RPM
- Signal Confidence: {physio.get('confidence', 'N/A')}

### Behavioral Readings
- Blink Rate: {behavioral.get('blink_rate', 'N/A')} blinks/min
- Eye Closure Ratio: {behavioral.get('eye_closure', 'N/A')}
- Gaze Stability: {behavioral.get('gaze_stability', 'N/A')}
- Head Pose: pitch={behavioral.get('head_pose', {}).get('pitch', 'N/A')}, yaw={behavioral.get('head_pose', {}).get('yaw', 'N/A')}, roll={behavioral.get('head_pose', {}).get('roll', 'N/A')}""")

        # Triage result
        if triage_result:
            context_parts.append(f"""
### AI Triage Verdict
- Verdict: {triage_result.verdict if hasattr(triage_result, 'verdict') else triage_result.get('verdict', 'N/A')}
- Confidence: {triage_result.confidence if hasattr(triage_result, 'confidence') else triage_result.get('confidence', 'N/A')}
- Summary: {triage_result.summary if hasattr(triage_result, 'summary') else triage_result.get('summary', 'N/A')}""")

            # Debate log
            debate = triage_result.debate_log if hasattr(triage_result, 'debate_log') else triage_result.get('debate_log', [])
            for arg in debate:
                name = arg.agent_name if hasattr(arg, 'agent_name') else arg.get('agent_name', '')
                assessment = arg.assessment if hasattr(arg, 'assessment') else arg.get('assessment', '')
                reasoning = arg.reasoning if hasattr(arg, 'reasoning') else arg.get('reasoning', '')
                context_parts.append(f"- {name}: {assessment} — {reasoning}")

            # Guidance
            guidance = triage_result.guidance if hasattr(triage_result, 'guidance') else triage_result.get('guidance')
            if guidance:
                context_parts.append(f"\n### Wellness Guidance\n{guidance}")

        # Wellness score
        if wellness_score:
            context_parts.append(f"""
### Wellness Score
- Score: {wellness_score.get('score', 'N/A')}/100 ({wellness_score.get('category', 'N/A')})""")

        # Emotional map
        if emotional_map:
            context_parts.append(f"""
### Emotional State
- {emotional_map.get('quadrant_label', 'N/A')}
- Arousal: {emotional_map.get('arousal', 'N/A')}, Valence: {emotional_map.get('valence', 'N/A')}
- {emotional_map.get('description', '')}""")

        # Contradiction
        if contradiction and contradiction.get("has_contradiction"):
            context_parts.append(f"""
### Signal Contradiction Detected
- Type: {contradiction.get('label', 'N/A')}
- {contradiction.get('explanation', '')}""")

        # Trend
        if trend_summary and trend_summary.get("has_enough_data"):
            context_parts.append(f"""
### Temporal Trends
- {trend_summary.get('trend_narrative', 'No trend data.')}""")
            if trend_summary.get("alerts"):
                for alert in trend_summary["alerts"]:
                    context_parts.append(f"- ALERT: {alert}")

        self._context = "\n".join(context_parts)

    def chat(self, user_message: str) -> str:
        """
        Send a message to the wellness chat and get a response.

        Parameters:
            user_message: The user's question or message.

        Returns:
            str: The assistant's response.
        """
        if not GROQ_API_KEY:
            return "Chat unavailable — GROQ_API_KEY not configured."

        # Add user message to history
        self.chat_history.append({"role": "user", "content": user_message})

        # Trim history if needed
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

        # Build messages array
        messages = [
            {"role": "system", "content": WELLNESS_CHAT_SYSTEM_PROMPT + "\n\n" + self._context},
        ]
        messages.extend(self.chat_history)

        try:
            client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.5,
                max_tokens=300,
            )
            assistant_msg = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Chat error: {e}")
            assistant_msg = "I'm having trouble connecting right now. Please try again in a moment."

        # Add assistant response to history
        self.chat_history.append({"role": "assistant", "content": assistant_msg})

        return assistant_msg

    def reset(self):
        """Clear chat history for a new session."""
        self.chat_history = []
        self._context = ""
