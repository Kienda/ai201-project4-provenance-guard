"""Provenance Guard Flask API.

Endpoints:

* ``GET  /``        - health check.
* ``POST /submit``  - classify submitted text (rate limited).
* ``POST /appeal``  - open an appeal against a prior classification.
* ``GET  /log``     - return the full audit trail.

Detection pipeline for ``POST /submit``::

    Groq detection -> Stylometric analysis -> Confidence calculator
                   -> Transparency label -> Audit log -> JSON response
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import audit
import confidence
import stylometry
from groq_detector import GroqDetectionError, detect
from labels import generate_label

app = Flask(__name__)

# In-memory rate limiter. No global default limits; limits are applied
# per-endpoint via the @limiter.limit decorator.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


def _utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _groq_p_ai(prediction: str, confidence_score: float) -> float:
    """Convert a Groq prediction/confidence pair into a P(AI) probability."""
    if prediction.strip().lower() == "ai":
        return confidence_score
    return 1.0 - confidence_score


@app.get("/")
def index() -> Any:
    """Simple health check endpoint."""
    return jsonify({"status": "running"})


@app.post("/submit")
@limiter.limit("10 per minute;100 per day")
def submit() -> Any:
    """Run the detection pipeline and record the result in the audit log."""
    # Parse JSON defensively so malformed bodies yield a clean 400.
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    text = data.get("text")
    creator_id = data.get("creator_id")

    if not text or not isinstance(text, str):
        return jsonify({"error": "Field 'text' is required"}), 400
    if not creator_id or not isinstance(creator_id, str):
        return jsonify({"error": "Field 'creator_id' is required"}), 400

    content_id = str(uuid.uuid4())

    # Signal 1: Groq large language model.
    try:
        groq_result = detect(text)
    except GroqDetectionError:
        return jsonify({"error": "Groq service unavailable"}), 500

    groq_p_ai = _groq_p_ai(groq_result["prediction"], groq_result["confidence"])

    # Signal 2: stylometric heuristics.
    stylometric_p_ai = stylometry.analyze(text)["stylometric_p_ai"]

    # Combine signals into a final prediction and confidence.
    scored = confidence.combine(groq_p_ai, stylometric_p_ai)
    prediction = scored["prediction"]
    overall_confidence = scored["confidence"]

    # Transparency label derived from the final prediction/confidence.
    label = generate_label(prediction, overall_confidence)

    # Persist a complete audit record for this submission.
    audit.save_log(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "timestamp": _utc_timestamp(),
            "prediction": prediction,
            "confidence": overall_confidence,
            "groq_score": scored["groq_score"],
            "stylometric_score": scored["stylometric_score"],
            "transparency_label": label["type"],
            "status": "classified",
            "appealed": False,
            "appeal_reasoning": None,
        }
    )

    return jsonify(
        {
            "content_id": content_id,
            "prediction": prediction,
            "confidence": overall_confidence,
            "label": label,
        }
    )


@app.post("/appeal")
def appeal() -> Any:
    """Open an appeal against a previous classification."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    content_id = data.get("content_id")
    creator_reasoning = data.get("creator_reasoning")

    if not content_id or not isinstance(content_id, str):
        return jsonify({"error": "Field 'content_id' is required"}), 400
    if not creator_reasoning or not isinstance(creator_reasoning, str):
        return jsonify({"error": "Field 'creator_reasoning' is required"}), 400

    updated = audit.update_log(
        content_id,
        {
            "status": "under_review",
            "appealed": True,
            "appeal_reasoning": creator_reasoning,
        },
    )
    if updated is None:
        return jsonify({"error": "content_id not found"}), 404

    return jsonify({"message": "Appeal received.", "status": "under_review"})


@app.get("/log")
def log() -> Any:
    """Return every recorded audit entry."""
    return jsonify({"entries": audit.get_logs()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
