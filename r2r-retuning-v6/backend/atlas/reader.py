"""Read the precomputed gain atlas.

`run_gain_atlas.py` writes one JSON per cell offline, because a live search is
impossible here: without a CUDA driver a twin gain-evaluation costs ~0.92 s, so
the paper's 2,805-evaluation search is ~43 minutes per cell.

This module only reads. It never computes a gain, never writes into the atlas,
and never interpolates: a protocol the atlas does not cover says so. The atlas
is written by a long-running job, so a partial atlas is the normal case and
every function here is written for it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ATLAS_DIR = PROJECT_ROOT / "data" / "gain_atlas"
CELL_DIR = ATLAS_DIR / "cells"

# The schema stamp every cell carries. `run_gain_atlas.py` imports this constant
# so writer and reader can never disagree about what "current" means.
#
# Bump it whenever a cell gains, loses or changes the meaning of a field. A cell
# written under an older version is treated as ABSENT here rather than served,
# because every consumer reads a missing field with optional chaining: an old
# cell lacking `ti_scale_on_bound` would render no bound note at all, silently
# presenting a boundary-pinned optimum as a free one. "Not precomputed" is the
# honest answer; the generator recomputes it on its next pass.
# v4: ET3M is now identified the way the paper defines it -- three records at
# v0 x {0.5, 1, 2}, seeds base + 17*i, ONE joint fit through the
# multi-condition cost (Eq. 11). v3 ran a single 51 s record at a single
# operating point, so its ET3M cells measured a different excitation.
# v5: the anti-alias cutoff is an AXIS, not a constant. The Twin Study screen
# offers LPF as a control, so holding it at 50 Hz meant every other value the
# user could type answered "not precomputed" -- the honest answer to a question
# the grid simply could not take.
ATLAS_VERSION = "gain_atlas_v5"

# The protocol values the atlas holds fixed. A request differing on any of
# these is off the grid, however close it looks.
# `record_s` is deliberately NOT here: it is not one fixed number but a function
# of the excitation (see `backend.pipeline.payloads.PAPER_RECORD_S`), and
# `on_grid` checks it against that table instead.
# The cutoffs the atlas holds, matched on both channels the way the paper's
# protocols do. `None` is a real setting -- unfiltered logging -- and is stored
# as the string "none" in the cell so JSON can carry it.
#
# 20 Hz is included deliberately even though the paper reports it as a
# feasibility failure ("most of its runs fail to converge", Fig. S6 caption).
# A cell that identifies and then fails is the paper's own answer to that
# cutoff, and is far more useful to a practitioner than "not precomputed".
LPF_AXIS_HZ: tuple[float | None, ...] = (None, 20.0, 50.0, 100.0, 200.0)


def lpf_key(value: Any) -> str:
    """Canonical cell/index spelling for a cutoff, including no filter."""
    if value is None or value == "" or str(value).lower() == "none":
        return "none"
    return f"{float(value):g}"


FIXED: dict[str, float | int] = {
    "pct_T": 0.003,
    "pct_v": 0.003,
    "Kp_star": 100.0,
    "seed": 0,
}


def load_manifest() -> dict[str, Any] | None:
    path = ATLAS_DIR / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _read(path: Path) -> dict[str, Any] | None:
    """A corrupt or half-written cell is skipped, never fatal.

    The generator writes atomically, but it is running while this reads, so
    treating an unreadable file as absent is the honest behaviour.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


@lru_cache(maxsize=1)
def twin_index() -> dict[tuple[str, float, str, str], dict[str, Any]]:
    """Index every twin cell by (plant_id, T_log_ms, excitation, LPF).

    Cached because the atlas has hundreds of cells; call `twin_index.cache_clear()`
    to pick up cells written since. The route clears it per request while the
    generator is still running, which costs a directory scan and is worth it.
    """
    index: dict[tuple[str, float, str, str], dict[str, Any]] = {}
    directory = CELL_DIR / "twin"
    if not directory.exists():
        return index
    for path in sorted(directory.glob("*.json")):
        cell = _read(path)
        if not cell:
            continue
        # A cell from an older atlas version is not a slightly-old answer, it
        # is a different schema. Dropping it here is what makes the protocol
        # read `not_precomputed` instead of rendering fields that no longer
        # mean what the screen thinks they mean.
        if cell.get("atlas_version") != ATLAS_VERSION:
            continue
        # A cell recorded at a record length the protocol no longer publishes
        # (ET1 at 7 s under dual-channel noise, before Table S1's 30 s rule was
        # honoured) answers for a different experiment: drop it, never serve it.
        if cell.get("record_s") is not None:
            from backend.pipeline.payloads import paper_record_s

            published = paper_record_s(str(cell.get("excitation", "")), cell.get("pct_v"))
            if published is not None and float(cell["record_s"]) != published:
                continue
        try:
            key = (
                str(cell["plant_id"]),
                float(cell["T_log_ms"]),
                str(cell["excitation"]),
                lpf_key(cell.get("LPF_T_hz")),
            )
        except (KeyError, TypeError, ValueError):
            continue
        index[key] = cell
    return index


def reference_for(plant_id: str) -> dict[str, Any] | None:
    return _read(CELL_DIR / "reference" / f"{plant_id}.json")


def cell_counts() -> dict[str, int]:
    """How many cells exist and are servable right now, per kind.

    Twin cells are counted through `twin_index`, so a cell dropped for a
    version mismatch is not counted as present: the number the screen shows is
    the number it could actually serve. Counted here rather than in the browser
    because the browser cannot see the atlas directory.
    """
    reference_dir = CELL_DIR / "reference"
    return {
        "twin": len(twin_index()),
        "reference": (
            len(list(reference_dir.glob("*.json"))) if reference_dir.exists() else 0
        ),
    }


def on_grid(protocol: Mapping[str, Any]) -> str | None:
    """Return the first fixed field the protocol differs on, or None."""
    from backend.pipeline.payloads import paper_record_s

    for field, expected in FIXED.items():
        actual = protocol.get(field)
        if actual is None or float(actual) != float(expected):
            return field
    # The record length is pinned per excitation, not globally: the atlas holds
    # each excitation at its own published duration, so a request that shortened
    # or lengthened one is off the grid even though every other field matches.
    published = paper_record_s(protocol.get("excitation", ""), protocol.get("pct_v"))
    actual = protocol.get("record_s")
    if published is None or actual is None or float(actual) != published:
        return "record_s"
    # The cutoff axis is matched on both channels, which is what every published
    # protocol does (field-matched 50/50, logging-only 100/100). A request that
    # decouples them is a real protocol the atlas simply does not hold.
    axis = {lpf_key(value) for value in LPF_AXIS_HZ}
    if lpf_key(protocol.get("LPF_T_hz")) not in axis:
        return "LPF_T_hz"
    if lpf_key(protocol.get("LPF_v_hz")) != lpf_key(protocol.get("LPF_T_hz")):
        return "LPF_v_hz"
    return None


def _ratio(achieved: float | None, reference: float | None) -> float | None:
    if achieved is None or reference is None or reference == 0:
        return None
    return achieved / reference


def lookup(plant_id: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one (plant, protocol) to an atlas payload.

    Statuses, all of which the UI must be able to render:
      ok                 cell and reference both present; both ratios computed
      reference_pending  the cell exists but the plant's reference has not been
                         computed yet -- normal while the generator runs
      not_converged      the identification under this protocol failed, so there
                         are no gains to score
      not_precomputed    the protocol is off-grid, or its cell does not exist
    """
    off_grid = on_grid(protocol)
    if off_grid is not None:
        return {"status": "not_precomputed", "off_grid_field": off_grid}

    try:
        key = (
            str(plant_id),
            float(protocol["T_log_ms"]),
            str(protocol["excitation"]),
            lpf_key(protocol.get("LPF_T_hz")),
        )
    except (KeyError, TypeError, ValueError):
        return {"status": "not_precomputed", "off_grid_field": "T_log_ms"}

    cell = twin_index().get(key)
    if cell is None:
        return {"status": "not_precomputed", "off_grid_field": None}

    if not cell.get("converged", False):
        return {
            "status": "not_converged",
            "cell": cell,
            "ratio_vs_full": None,
            "ratio_vs_coarse": None,
            "reference": reference_for(plant_id),
        }

    reference = reference_for(plant_id)
    achieved = cell.get("S_achieved_on_physical")
    if reference is None:
        return {
            "status": "reference_pending",
            "cell": cell,
            "reference": None,
            "ratio_vs_full": None,
            "ratio_vs_coarse": None,
        }

    return {
        "status": "ok",
        "cell": cell,
        "reference": reference,
        "ratio_vs_full": _ratio(achieved, (reference.get("full") or {}).get("S")),
        "ratio_vs_coarse": _ratio(achieved, (reference.get("coarse") or {}).get("S")),
    }
