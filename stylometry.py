"""Lightweight stylometric detection signal for Provenance Guard.

This is the second detection signal in the pipeline. It inspects surface-level
writing features (sentence-length uniformity, contraction usage, lexical
diversity, average word length) and returns a probability that the text was
AI-generated.

The heuristics are intentionally simple and deterministic; they complement the
Groq signal rather than replace it.
"""

from __future__ import annotations

import re
import statistics
from typing import List, TypedDict

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_RE = re.compile(r"[.!?]+")

# Common English contractions are a strong human-writing tell.
_CONTRACTIONS = (
    "n't",
    "'re",
    "'ve",
    "'ll",
    "'d",
    "'m",
    "'s",
)


class StylometryResult(TypedDict):
    """Shape of the value returned by :func:`analyze`."""

    stylometric_p_ai: float


def _clamp(value: float) -> float:
    """Constrain a value to the closed interval [0, 1]."""
    return max(0.0, min(1.0, value))


def _sentence_uniformity_score(sentence_word_counts: List[int]) -> float:
    """Higher when sentence lengths are unusually uniform (AI-like).

    Human writing tends to vary sentence length; AI output is often more even.
    """
    if len(sentence_word_counts) < 2:
        return 0.5

    mean = statistics.fmean(sentence_word_counts)
    if mean == 0:
        return 0.5

    # Coefficient of variation: low spread -> AI-leaning.
    cv = statistics.pstdev(sentence_word_counts) / mean
    # cv ~0 -> ~1.0 (AI); cv >= 0.6 -> ~0.0 (human).
    return _clamp(1.0 - (cv / 0.6))


def _contraction_score(words: List[str]) -> float:
    """Higher when contractions are absent (AI-like)."""
    contraction_count = sum(
        1 for word in words if any(word.endswith(suffix) for suffix in _CONTRACTIONS)
    )
    ratio = contraction_count / len(words)
    # No contractions -> AI-leaning; ratio >= 0.05 -> human-leaning.
    return _clamp(1.0 - (ratio / 0.05))


def _lexical_diversity_score(words: List[str]) -> float:
    """Higher when lexical diversity is low (AI-like repetition)."""
    type_token_ratio = len(set(words)) / len(words)
    # ttr <= 0.4 -> AI-leaning; ttr >= 0.7 -> human-leaning.
    return _clamp((0.7 - type_token_ratio) / 0.3)


def _word_length_score(words: List[str]) -> float:
    """Higher when average word length is elevated (AI-like formality)."""
    avg_len = statistics.fmean(len(word) for word in words)
    # avg <= 4.2 -> human-leaning; avg >= 5.4 -> AI-leaning.
    return _clamp((avg_len - 4.2) / 1.2)


def analyze(text: str) -> StylometryResult:
    """Estimate the probability that ``text`` was AI-generated.

    Args:
        text: The submitted content to analyze.

    Returns:
        A dictionary with ``stylometric_p_ai`` in the range [0, 1], where
        higher values indicate stronger AI-leaning stylometric signals.
    """
    words = _WORD_RE.findall(text.lower())
    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]

    # Not enough signal to say anything useful -> stay neutral.
    if not words or not sentences:
        return {"stylometric_p_ai": 0.5}

    sentence_word_counts = [len(_WORD_RE.findall(s)) for s in sentences]

    # Average the individual feature scores into a single probability.
    feature_scores = [
        _sentence_uniformity_score(sentence_word_counts),
        _contraction_score(words),
        _lexical_diversity_score(words),
        _word_length_score(words),
    ]
    p_ai = statistics.fmean(feature_scores)

    return {"stylometric_p_ai": round(_clamp(p_ai), 4)}
