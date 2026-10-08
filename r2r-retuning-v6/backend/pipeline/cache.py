"""A content-addressed result cache keyed by the RunSpec hash.

Identify runs in about three seconds, so this is not here for speed. It is
here so that the section 4 job runner and the existing `prefer_cache` studies
share one cache seam instead of growing a second one.

A corrupt or unreadable entry is a miss, never an error: the cache is an
optimisation and must never be the reason a request fails.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "session" / "cache"


def _safe_kind(kind: str) -> str:
    """Sanitise a cache kind (namespace) to prevent path traversal."""
    safe = "".join(char for char in kind if char.isalnum() or char in "-_")
    if not safe:
        raise ValueError(f"invalid cache kind {kind!r}")
    return safe


def cache_path(kind: str, key: str) -> Path:
    safe_kind = _safe_kind(kind)
    safe_key = "".join(char for char in key if char in "0123456789abcdef")
    if not safe_key:
        raise ValueError(f"invalid cache address {kind!r}/{key!r}")
    return CACHE_DIR / safe_kind / f"{safe_key}.json"


def load(kind: str, key: str) -> dict | None:
    path = cache_path(kind, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def store(kind: str, key: str, payload: dict) -> Path:
    path = cache_path(kind, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use unique temp name to avoid collisions between concurrent writers
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def clear(kind: str | None = None) -> int:
    if kind is None:
        root = CACHE_DIR
    else:
        root = CACHE_DIR / _safe_kind(kind)
    if not root.exists():
        return 0
    removed = 0
    # Remove both cache files and orphaned temp files
    for path in root.rglob("*.json"):
        path.unlink()
        removed += 1
    for path in root.rglob("*.tmp"):
        path.unlink()
    return removed
