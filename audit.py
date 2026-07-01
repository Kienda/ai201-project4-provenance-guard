"""Structured audit logging for Provenance Guard (Milestone 3).

Entries are persisted to a JSON file on disk. This is a temporary store for
Milestone 3; a database (e.g. SQLite) is planned for a later milestone.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

# Path to the JSON audit log, resolved relative to this file so the location
# is stable regardless of the current working directory.
AUDIT_LOG_PATH: str = os.path.join(os.path.dirname(__file__), "audit_log.json")

AuditEntry = Dict[str, Any]


def _read_all() -> List[AuditEntry]:
    """Return every stored entry, tolerating a missing or empty file."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return []

    try:
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        # A corrupt or unreadable log should not crash the API; start fresh.
        return []

    return data if isinstance(data, list) else []


def save_log(entry: AuditEntry) -> None:
    """Append a single audit ``entry`` to the JSON log.

    The log file is created automatically if it does not yet exist.

    Args:
        entry: A fully-formed audit record to persist.
    """
    entries = _read_all()
    entries.append(entry)

    with open(AUDIT_LOG_PATH, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)


def get_logs() -> List[AuditEntry]:
    """Return all audit entries recorded so far."""
    return _read_all()
