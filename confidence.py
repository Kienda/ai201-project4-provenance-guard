"""Confidence calculator for Provenance Guard.

Combines the individual detection signals (Groq + stylometry) into a single
prediction and confidence score. Each signal is expressed as ``p_ai`` -- the
probability that the text is AI-generated -- so they can be blended with a
simple weighted average.
"""

from __future__ import annotations

from typing import TypedDict

# Relative trust placed in each signal. Groq is the stronger signal, so it
# carries the larger weight. Weights must sum to 1.0.
GROQ_WEIGHT: float = 0.7
STYLOMETRIC_WEIGHT: float = 0.3


class ConfidenceResult(TypedDict):
    """Combined prediction plus per-signal scores aligned to that prediction."""

    prediction: str
    confidence: float
    groq_score: float
    stylometric_score: float


def combine(groq_p_ai: float, stylometric_p_ai: float) -> ConfidenceResult:
    """Blend the two signals into a final prediction and confidence.

    Args:
        groq_p_ai: Groq's probability that the text is AI-generated [0, 1].
        stylometric_p_ai: Stylometry's probability of AI generation [0, 1].

    Returns:
        The combined ``prediction`` ("Likely AI" / "Likely Human"), the overall
        ``confidence`` in that prediction, and each signal's score expressed as
        confidence toward the winning prediction.
    """
    combined_p_ai = GROQ_WEIGHT * groq_p_ai + STYLOMETRIC_WEIGHT * stylometric_p_ai

    if combined_p_ai >= 0.5:
        prediction = "Likely AI"
        confidence = combined_p_ai
        groq_score = groq_p_ai
        stylometric_score = stylometric_p_ai
    else:
        # Re-express confidence and per-signal scores toward "Human".
        prediction = "Likely Human"
        confidence = 1.0 - combined_p_ai
        groq_score = 1.0 - groq_p_ai
        stylometric_score = 1.0 - stylometric_p_ai

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "groq_score": round(groq_score, 2),
        "stylometric_score": round(stylometric_score, 2),
    }
