"""The single-session file store.

One session, persisted to data/session/current.json so a browser reload or a
server restart does not lose the user's selection. A corrupt or unreadable
file yields a default session rather than an error: losing a selection is an
inconvenience, refusing to start is a fault.
"""

from __future__ import annotations

import json
from pathlib import Path

from .state import SessionState, from_dict, to_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_PATH = PROJECT_ROOT / "data" / "session" / "current.json"


def load() -> SessionState:
    if not SESSION_PATH.exists():
        return SessionState()
    try:
        payload = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return SessionState()
        return from_dict(payload)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return SessionState()


def save(state: SessionState) -> Path:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SESSION_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(to_dict(state), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(SESSION_PATH)
    return SESSION_PATH


def reset() -> SessionState:
    state = SessionState()
    save(state)
    return state
