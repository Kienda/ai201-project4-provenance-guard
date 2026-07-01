"""Provenance Guard Flask API (Milestone 3).

Exposes three endpoints:

* ``GET  /``        - health check.
* ``POST /submit``  - classify submitted text as human- or AI-written.
* ``GET  /log``     - return the audit trail of past classifications.

Milestone 3 uses a single detection signal (Groq) and writes a structured
audit log to a JSON file. Confidence combination, stylometry, appeals, and
transparency labels are reserved for later milestones.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from flask import Flask, jsonify, request

import audit
from groq_detector import GroqDetectionError, detect

app = Flask(__name__)

# Placeholder label; real transparency labels arrive in Milestone 4.
PLACEHOLDER_LABEL: str = "Placeholder label (Milestone 4)"


def _utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _attribution_from_prediction(prediction: str) -> str:
    """Map a raw model prediction to a human-readable attribution."""
    return "Likely AI" if prediction.strip().lower() == "ai" else "Likely Human"


@app.get("/")
def index() -> Any:
    """Simple health check endpoint."""
    return jsonify({"status": "running"})


@app.post("/submit")
def submit() -> Any:
    """Classify submitted text and record the result in the audit log."""
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

    # First (and only, for Milestone 3) detection signal.
    try:
        result = detect(text)
    except GroqDetectionError:
        return jsonify({"error": "Groq service unavailable"}), 500

    attribution = _attribution_from_prediction(result["prediction"])
    confidence = result["confidence"]

    entry: Dict[str, Any] = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": _utc_timestamp(),
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": confidence,
        "status": "classified",
    }
    audit.save_log(entry)

    return jsonify(
        {
            "content_id": content_id,
            "attribution": attribution,
            "confidence": confidence,
            "label": PLACEHOLDER_LABEL,
            "status": "classified",
        }
    )


@app.get("/log")
def log() -> Any:
    """Return every recorded audit entry."""
    return jsonify({"entries": audit.get_logs()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
