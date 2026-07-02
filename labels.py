"""Transparency labels for Provenance Guard (Milestone 5).

Turns a final prediction and confidence into a user-facing transparency label.
Exactly one of three labels is returned for any input.
"""

from __future__ import annotations

from typing import TypedDict

# Confidence at or above this threshold counts as "high confidence".
HIGH_CONFIDENCE_THRESHOLD: float = 0.90


class TransparencyLabel(TypedDict):
    """A transparency label: a stable ``type`` plus user-facing ``text``."""

    type: str
    text: str


def generate_label(prediction: str, confidence: float) -> TransparencyLabel:
    """Return the transparency label for a prediction/confidence pair.

    Args:
        prediction: The final attribution ("Likely AI" or "Likely Human").
        confidence: Overall confidence in the prediction [0, 1].

    Returns:
        One of three labels: high-confidence AI, high-confidence human, or
        uncertain attribution.
    """
    is_high_confidence = confidence >= HIGH_CONFIDENCE_THRESHOLD

    if prediction == "Likely AI" and is_high_confidence:
        return {
            "type": "high_confidence_ai",
            "text": (
                "Likely AI-generated. Our analysis strongly suggests this "
                "content was generated using AI. Creators may submit an appeal."
            ),
        }

    if prediction == "Likely Human" and is_high_confidence:
        return {
            "type": "high_confidence_human",
            "text": (
                "Likely Human-written. Our analysis strongly suggests this "
                "content was written by a human author."
            ),
        }

    return {
        "type": "uncertain",
        "text": (
            "Uncertain Attribution. Our system could not confidently determine "
            "whether this content was AI-generated or human-written."
        ),
    }
