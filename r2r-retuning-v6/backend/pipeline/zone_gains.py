"""The saved per-zone gains: the six numbers a plant was commissioned with.

The adopted two-stage method (user, 2026-10-07): a free (K_p*, T_I) per tension zone
is found once by the 6-D search and saved here; every later case of the same plant
freezes UW and RW at these values and retunes only the out-feeder. The key is the
PLANT (preset or hand-edited parameters) and the cost tier -- not the drift, not the
protocol -- because the next case is the same line after it has drifted.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

from .cache import CACHE_DIR
from .payloads import RunSpec, to_dict

STORE_DIR = CACHE_DIR.parent / "zone_gains"


def key(run: RunSpec, cost_tier: str) -> str:
    blob = json.dumps({"plant": to_dict(run.plant), "tier": cost_tier}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _path(run: RunSpec, cost_tier: str) -> Path:
    return Path(STORE_DIR) / f"{key(run, cost_tier)}.json"


def load(run: RunSpec, cost_tier: str) -> dict[str, Any] | None:
    path = _path(run, cost_tier)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        # an unreadable record is no record: the next retune searches the six again
        return None
    ok = isinstance(record, dict) and len(record.get("kp_star_per_zone") or []) == 3 \
        and len(record.get("ti_s_per_zone") or []) == 3
    return record if ok else None


def save(run: RunSpec, cost_tier: str, gains: Mapping[str, Any], *, source_run_hash: str,
         S_twin: float | None = None) -> dict[str, Any]:
    kp = [float(v) for v in gains["kp_star_per_zone"]]
    ti = [float(v) for v in gains["ti_s_per_zone"]]
    if len(kp) != 3 or len(ti) != 3 or not all(math.isfinite(v) and v > 0 for v in kp + ti):
        raise ValueError("per-zone gains must be three positive K_p* and three positive T_I")
    record = {
        "plant": to_dict(run.plant),
        "cost_tier": cost_tier,
        "kp_star_per_zone": kp,
        "ti_s_per_zone": ti,
        "zones": ["UW", "OutFeeder", "RW"],
        "source_run_hash": source_run_hash,
        "S_twin": S_twin,
        "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    path = _path(run, cost_tier)
    path.parent.mkdir(parents=True, exist_ok=True)
    # a temp name per writer: two retunes of one plant may save at once
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return record


def delete(run: RunSpec, cost_tier: str) -> bool:
    path = _path(run, cost_tier)
    if not path.exists():
        return False
    path.unlink()
    return True
