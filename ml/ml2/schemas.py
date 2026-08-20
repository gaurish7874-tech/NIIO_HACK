"""
schemas.py
----------
Pydantic data models for the ML-2 Triage and RAG Guidance pipeline.
Defines structured types for agent arguments, judge decisions, and final results.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class TriageVerdict(str, Enum):
    """Possible triage outcomes from the Judge Agent."""
    NORMAL = "normal"
    MILD_STRESS = "mild_stress"
    HIGH_STRESS = "high_stress"
    FATIGUE = "fatigue"
    DROWSINESS = "drowsiness"
    ANXIETY = "anxiety"
    NEEDS_ATTENTION = "needs_attention"


class AgentArgument(BaseModel):
    """Structured output from the Physio Agent or Behavioral Agent."""
    agent_name: str = Field(description="Name of the agent (e.g., 'Physio Agent')")
    assessment: str = Field(description="Brief assessment label (e.g., 'elevated stress indicators')")
    reasoning: str = Field(description="2-3 sentence explanation of the assessment")
    confidence: float = Field(ge=0.0, le=1.0, description="Agent's confidence in its assessment")
    key_signals: list[str] = Field(
        default_factory=list,
        description="Key signal readings that support the assessment (e.g., 'HR 95 BPM (elevated)')"
    )


class JudgeDecision(BaseModel):
    """Structured output from the Judge Agent after evaluating both agent arguments."""
    verdict: TriageVerdict = Field(description="Final triage verdict")
    confidence: float = Field(ge=0.0, le=1.0, description="Judge's overall confidence")
    summary: str = Field(description="1-2 sentence plain-English summary for the user")
    needs_clarification: bool = Field(
        default=False,
        description="Whether the Judge needs more info from the user"
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description="Question to ask the user if needs_clarification is True"
    )
    physio_weight: float = Field(
        ge=0.0, le=1.0, default=0.5,
        description="How much the physio signals influenced the verdict"
    )
    behavioral_weight: float = Field(
        ge=0.0, le=1.0, default=0.5,
        description="How much the behavioral signals influenced the verdict"
    )


class TriageResult(BaseModel):
    """Complete output of the ML-2 Triage pipeline — returned to the backend."""
    verdict: TriageVerdict
    confidence: float
    summary: str
    debate_log: list[AgentArgument] = Field(
        default_factory=list,
        description="The Physio and Behavioral agent arguments (the 'debate')"
    )
    judge_decision: JudgeDecision
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    guidance: Optional[str] = Field(
        default=None,
        description="RAG-generated wellness guidance (filled after triage)"
    )
    guidance_sources: list[str] = Field(
        default_factory=list,
        description="Source document names from the knowledge base"
    )

    # --- New: Wellness Score ---
    wellness_score: Optional[dict] = Field(
        default=None,
        description="Composite wellness score (0-100) with breakdown and category"
    )

    # --- New: Cross-Modal Contradiction ---
    contradiction: Optional[dict] = Field(
        default=None,
        description="Cross-modal contradiction detection result"
    )

    # --- New: Arousal-Valence Emotional Map ---
    emotional_map: Optional[dict] = Field(
        default=None,
        description="Russell's Circumplex arousal-valence mapping"
    )

    # --- New: Temporal Trend Summary ---
    trend_summary: Optional[dict] = Field(
        default=None,
        description="Temporal trend analysis from the wellness timeline"
    )

    # --- New: Baseline Deviation ---
    baseline_deviation: Optional[dict] = Field(
        default=None,
        description="How current readings deviate from user's personal baseline"
    )


class GuidanceResult(BaseModel):
    """Output from the RAG Guidance Layer."""
    advice: str = Field(description="Grounded wellness advice based on retrieved documents")
    sources: list[str] = Field(
        default_factory=list,
        description="Names of source documents used to generate the advice"
    )
    disclaimer: str = Field(
        default="This is not medical advice. Please consult a healthcare professional for medical concerns.",
        description="Standard disclaimer"
    )
