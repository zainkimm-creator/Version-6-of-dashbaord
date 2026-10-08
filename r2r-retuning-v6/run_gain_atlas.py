#!/usr/bin/env python
"""Precompute the gain atlas the Twin Study screen reads.

The question the atlas answers, per acquisition protocol:

    "If I identify the plant under THIS protocol and tune from the resulting
     digital twin, are the gains as good as the ones I would have found if I
     had known the real plant?"

Answering it live was thought impossible on this machine, because the scalar
NumPy cost path costs ~0.92 s per gain evaluation (there is no CUDA driver
here). It is not: supplement S8.1 states the JAX vmap+JIT simulator finishes the
full sweep "in tens of seconds on a multi-core CPU", and it does -- 2,805
evaluations in ~3.5 s on this box, ~650x the scalar path. This script therefore
runs the paper's own HGS budget on every cell, not a reduced grid.

THE EVALUATION EPISODE. The cost S of Eq. (12) is measured on a STEP RESPONSE,
not on the 16 s E_Toggle record. Main text 2.5 calls its three quantities
"step-response metrics", normalises OS and t_s per channel by that channel's own
|dT_ref,i|, and times t_s from "the reference step" (singular); Algorithm 1
Steps 5 and 7 say "validation step response" and "step-response test"; and the
data package labels the 16 s E_Toggle record the SysID *identification* protocol
(excitation_schedules_v5_summary.md, campaign group C_retuning_field_matched).
Scoring that record instead scores 0 of 6 pre-registered anchors and inflates the
median cost ~7x; the stepseq episode below scores 5 of 6. See
docs/paper-v5.1-vs-dashboard-2026-09-07.md for the full argument and evidence.

Three quantities per plant, and one per protocol cell:

  reference_full    full 2,805-evaluation HGS on theta_true -> the plant's real
                    optimum. The yardstick, at the paper's exact budget.
  reference_coarse  the 10x10 log grid, also on theta_true. Retained so a cell
                    still carries both readings of the headline ratio and the
                    stored schema is unchanged; with the twins now searched at
                    the full budget, S_achieved/S_ref_full is the comparable one.
  achieved          identify under the protocol -> twin -> FULL HGS on the twin
                    -> transfer those gains to the PHYSICAL plant and score them
                    there, in one evaluation. This is the paper's HGS-only arm.

Gain transfer is in physical units, not search units. The search parameterises
T_I as a multiple of the plant's own auto_Ti, and the twin's auto_Ti differs
from the physical plant's, so a scale transferred directly would be a different
T_I. The twin's scale is resolved against the twin's auto_Ti before the gains
are scored on the physical plant.

Resumable by design: an 18-hour job must survive an interruption, so every cell
is written as its own file and an existing file is skipped. Kill it, reboot,
restart it -- at most the cell in flight is lost. The skip is version-aware: a
cell stamped with an older ATLAS_VERSION is recomputed rather than kept, because
the reader treats such a cell as absent and would otherwise never serve it.

Usage:
    .venv/bin/python run_gain_atlas.py                # run everything
    .venv/bin/python run_gain_atlas.py --plants P01   # one plant
    .venv/bin/python run_gain_atlas.py --dry-run      # print the plan and exit
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# One definition of "current", shared with the dashboard's reader. If the writer
# stamped a version the reader did not recognise, the cell would be written and
# then silently never served.
from backend.atlas.reader import ATLAS_VERSION, LPF_AXIS_HZ, lpf_key  # noqa: E402

ATLAS_DIR = PROJECT_ROOT / "data" / "gain_atlas"
CELL_DIR = ATLAS_DIR / "cells"

# The grid. Noise, LPF, record length, SysID gain and seed are fixed at the
# paper's field-matched values; T_log and excitation are the axes.
TLOG_MS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
EXCITATIONS = ("ET1", "ET3", "ET6", "ET3M", "E_Toggle", "EV1")
PCT_T = 0.003
PCT_V = 0.003
# The anti-alias cutoff is an axis, matched on both channels the way every
# published protocol sets it (field-matched 50/50, logging-only 100/100). The
# axis itself lives in the reader so writer and reader cannot disagree about
# which cutoffs exist.
KP_STAR_SYSID = 100.0
SEED = 0

# Coarse search: a 10x10 log grid over the campaign's own gain box. Kept as a
# second reading of the reference optimum; the twins now get the full budget.
COARSE_SIDE = 10

# The reading of Eq. (12)'s evaluation episode, and the only one consistent with
# both the paper's per-channel normalisers and its published magnitudes:
# three +20 % steps, ONE CHANNEL PER 5 s EPISODE, each held, edges at 1/6/11 s.
# "clean" = no sensor noise in the evaluation loop, "true" = metrics formed from
# true tensions, "clamp" = the outer-loop integral clamp the dashboard applies.
# Scored 5/6 on the pre-registered anchors (reports/anchor_sweep_stepseq.json);
# the previous default, "record", scored 0/6 (reports/anchor_sweep.json).
EVAL_MODEL_KEY = "stepseq-clean-true-clamp"


def _install_eval_model() -> None:
    """Point every cost evaluation in this run at the step-response episode."""
    import backend.validation.retuning as _R

    _R.EVAL_MODEL_OVERRIDE = EVAL_MODEL_KEY


def _costs(params, line_speed_m_s: float, ti_reference_s: float):
    """(scalar_cost, batch_cost) on the JAX kernel, under EVAL_MODEL_KEY.

    batch_cost scores a whole array of candidates in one dispatch; that single
    dispatch is what makes the paper's 2,805-point budget affordable per cell.
    """
    from backend.validation.retuning import make_jax_cost_function

    return make_jax_cost_function(params, line_speed_m_s, ti_reference_s)


def _log_grid(lo: float, hi: float, n: int) -> list[float]:
    """n points geometrically spaced over [lo, hi] -- both gains span decades."""
    if n < 2:
        return [math.sqrt(lo * hi)]
    step = (math.log(hi) - math.log(lo)) / (n - 1)
    return [math.exp(math.log(lo) + i * step) for i in range(n)]


def coarse_search(cost, kp_bounds, ti_bounds, side: int = COARSE_SIDE,
                  batch_cost=None):
    """Exhaustive side x side log grid. Returns (kp_star, ti_scale, S, evals)."""
    points = [(kp, ti)
              for kp in _log_grid(kp_bounds[0], kp_bounds[1], side)
              for ti in _log_grid(ti_bounds[0], ti_bounds[1], side)]
    if batch_cost is not None:
        values = batch_cost([p[0] for p in points], [p[1] for p in points])
        scored = [(float(v), kp, ti) for (kp, ti), v in zip(points, values)]
    else:
        scored = [(float(cost(kp, ti)), kp, ti) for kp, ti in points]
    S, kp, ti = min(scored, key=lambda row: row[0])
    return kp, ti, S, len(points)


def on_bound(value: float, bounds: tuple[float, float]) -> str | None:
    """Name the boundary an optimum landed on, or None if it is interior.

    Recorded per cell rather than asserted globally. Under Eq. (12) the T_I
    optimum pins at the box's upper edge -- the cost falls monotonically as
    integral action is removed -- so every consumer of this atlas must be able
    to say so for the cell in front of it instead of quietly presenting a
    boundary value as a free optimum.
    """
    if math.isclose(value, bounds[0], rel_tol=1e-9):
        return "lower"
    if math.isclose(value, bounds[1], rel_tol=1e-9):
        return "upper"
    return None


def cell_path(kind: str, key: str) -> Path:
    return CELL_DIR / kind / f"{key}.json"


def write_cell(kind: str, key: str, payload: dict[str, Any]) -> Path:
    path = cell_path(kind, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def cell_is_current(kind: str, key: str) -> bool:
    """True only if the cell exists AND was written by THIS atlas version.

    Skipping on existence alone is the loop that strands a cell: one written
    before a field existed is never recomputed, and because every consumer
    reads a missing field with optional chaining it renders as though the field
    were simply not applicable -- a boundary-pinned optimum shown as a free one.
    The reader drops a version-mismatched cell for the same reason, so a cell
    this function declines to skip is one the dashboard could not serve anyway.

    An unreadable or half-written file counts as not current, so the resumable
    run rewrites it rather than leaving it broken forever.
    """
    try:
        payload = json.loads(cell_path(kind, key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("atlas_version") == ATLAS_VERSION


def log(message: str) -> None:
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def build_reference(plant_id: str, force: bool) -> dict[str, Any] | None:
    """Both reference searches on theta_true for one plant."""
    from backend.validation.plants import parameters_for_plant
    from backend.validation.retuning import (
        KP_BOUNDS,
        TI_SCALE_BOUNDS,
        hierarchical_grid_search,
        plant_auto_ti_s,
    )

    key = plant_id
    if not force and cell_is_current("reference", key):
        log(f"reference {plant_id}: already present, skipping")
        return None

    params, meta = parameters_for_plant(plant_id)
    v0 = float(meta["v_ref_m_s"])
    auto_ti = plant_auto_ti_s(params, v0)
    cost, batch = _costs(params, v0, auto_ti)

    t0 = time.time()
    kp_c, ti_c, S_coarse, n_coarse = coarse_search(
        cost, KP_BOUNDS, TI_SCALE_BOUNDS, batch_cost=batch)
    t_coarse = time.time() - t0
    log(f"reference {plant_id}: coarse S={S_coarse:.4f} in {t_coarse:.1f} s")

    t0 = time.time()
    hgs = hierarchical_grid_search(cost, seed=SEED, batch_cost=batch)
    t_full = time.time() - t0
    log(
        f"reference {plant_id}: FULL S={hgs.best_twin_cost:.4f} "
        f"({hgs.twin_evaluations} evals) in {t_full:.1f} s"
    )

    payload = {
        "atlas_version": ATLAS_VERSION,
        "kind": "reference",
        "eval_model": EVAL_MODEL_KEY,
        "plant_id": plant_id,
        "auto_ti_s": auto_ti,
        "line_speed_m_s": v0,
        "full": {
            "kp_star": hgs.best_kp,
            "ti_scale": hgs.best_ti_scale,
            "ti_s": hgs.best_ti_scale * auto_ti,
            "kp_on_bound": on_bound(hgs.best_kp, KP_BOUNDS),
            "ti_scale_on_bound": on_bound(hgs.best_ti_scale, TI_SCALE_BOUNDS),
            "S": hgs.best_twin_cost,
            "evaluations": hgs.twin_evaluations,
            "search": "hierarchical_grid_search (paper budget)",
            "seconds": t_full,
        },
        "coarse": {
            "kp_star": kp_c,
            "ti_scale": ti_c,
            "ti_s": ti_c * auto_ti,
            "kp_on_bound": on_bound(kp_c, KP_BOUNDS),
            "ti_scale_on_bound": on_bound(ti_c, TI_SCALE_BOUNDS),
            "S": S_coarse,
            "evaluations": n_coarse,
            "search": f"{COARSE_SIDE}x{COARSE_SIDE} log grid",
            "seconds": t_coarse,
        },
        "produced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_cell("reference", key, payload)
    return payload


def build_twin_cell(plant_id: str, tlog_ms: float, excitation: str,
                    lpf_hz: float | None, force: bool):
    """Identify under one protocol, search the twin, score on the physical plant."""
    from backend.pipeline.identify import identify
    from backend.pipeline.payloads import (
        DriftSpec,
        PlantSpec,
        ProtocolSpec,
        RunSpec,
        paper_record_s,
        run_hash,
    )
    from backend.pipeline.plant import params_from_spec
    from backend.pipeline.twin import plant_spec_from_theta
    from backend.validation.retuning import (
        KP_BOUNDS,
        TI_SCALE_BOUNDS,
        hierarchical_grid_search,
        plant_auto_ti_s,
    )

    # Each excitation runs for its OWN published duration. Forcing one 16 s
    # window on all six truncates ET3/ET6/ET3M to identical schedules -- inside
    # 16 s all three carry only span1@2s, span2@7s, span3@12s -- so three of the
    # six axis values would return bit-identical cells.
    record_s = paper_record_s(excitation, PCT_V)
    if record_s is None:
        raise ValueError(f"no published record length for excitation {excitation!r}")

    plant_spec = PlantSpec(preset_id=plant_id)
    protocol = ProtocolSpec(
        T_log_ms=tlog_ms,
        excitation=excitation,
        record_s=record_s,
        pct_T=PCT_T,
        pct_v=PCT_V,
        LPF_T_hz=lpf_hz,
        LPF_v_hz=lpf_hz,
        Kp_star=KP_STAR_SYSID,
        seed=SEED,
    )
    run = RunSpec(plant_spec, DriftSpec(), protocol)
    key = run_hash(run)
    if not force and cell_is_current("twin", key):
        return None

    label = f"{plant_id} Tlog={tlog_ms:g}ms {excitation} LPF={lpf_key(lpf_hz)}"
    t0 = time.time()

    # A plant that DIVERGES under this protocol is a legitimate outcome, not a
    # crash. The LPF axis reaches cutoffs the paper itself calls unusable -- an
    # unfiltered or 20 Hz loop can destabilise -- and the simulator quite
    # correctly refuses to carry a non-finite state. Recording that as a
    # non-converged cell is the same answer the estimator's own non-convergence
    # gets, and keeps a 2,100-cell sweep from dying on one bad combination.
    try:
        result = identify(run)
        failure = None
    except (ValueError, RuntimeError, OverflowError, FloatingPointError) as exc:
        result = {"converged": False}
        failure = f"{type(exc).__name__}: {exc}"

    if not result["converged"]:
        payload = {
            "atlas_version": ATLAS_VERSION,
            "kind": "twin",
            "run_hash": key,
            "plant_id": plant_id,
            "T_log_ms": tlog_ms,
            "excitation": excitation,
            "LPF_T_hz": lpf_hz,
            "LPF_v_hz": lpf_hz,
            # The acquisition must be recorded on a failed cell too, or the
            # reader cannot tell a failure at the published record length from
            # one at a record length that is no longer published.
            "record_s": record_s,
            "pct_T": PCT_T,
            "pct_v": PCT_V,
            "converged": False,
            "MARE_theta_pct": None,
            "note": (f"simulation diverged under this protocol; no estimate ({failure})"
                     if failure else
                     "identification did not converge; no gains derived"),
            "failure": failure,
            "produced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        write_cell("twin", key, payload)
        log(f"{label}: FAILED -- {'diverged' if failure else 'no convergence'}")
        return payload

    physical, meta = params_from_spec(plant_spec)
    v0 = float(meta["v_ref_m_s"])

    twin_spec = plant_spec_from_theta(plant_spec, result["estimates"])
    twin_params, _ = params_from_spec(twin_spec)

    # Search the twin in the twin's own T_I units, at the paper's own HGS budget
    # -- this cell IS the paper's HGS-only arm, so it gets the paper's search.
    twin_auto_ti = plant_auto_ti_s(twin_params, v0)
    twin_cost, twin_batch = _costs(twin_params, v0, twin_auto_ti)
    hgs = hierarchical_grid_search(twin_cost, seed=SEED, batch_cost=twin_batch)
    kp, ti_scale, S_twin = hgs.best_kp, hgs.best_ti_scale, hgs.best_twin_cost
    n_evals = hgs.twin_evaluations

    # Transfer in PHYSICAL units: resolve the scale against the twin's auto_Ti,
    # then score that absolute T_I on the physical plant -- on the same episode,
    # so the achieved cost and the reference cost are the same quantity.
    ti_seconds = ti_scale * twin_auto_ti
    physical_auto_ti = plant_auto_ti_s(physical, v0)
    physical_cost, _ = _costs(physical, v0, physical_auto_ti)
    S_achieved = float(physical_cost(kp, ti_seconds / physical_auto_ti))

    elapsed = time.time() - t0
    payload = {
        "atlas_version": ATLAS_VERSION,
        "kind": "twin",
        "run_hash": key,
        "plant_id": plant_id,
        "T_log_ms": tlog_ms,
        "excitation": excitation,
        "pct_T": PCT_T,
        "pct_v": PCT_V,
        "LPF_T_hz": lpf_hz,
        "LPF_v_hz": lpf_hz,
        "record_s": record_s,
        "Kp_star_sysid": KP_STAR_SYSID,
        "seed": SEED,
        "converged": True,
        "MARE_theta_pct": result["MARE_theta_pct"],
        "twin_auto_ti_s": twin_auto_ti,
        "gains": {
            "kp_star": kp,
            "ti_scale_on_twin": ti_scale,
            "ti_s": ti_seconds,
            "kp_on_bound": on_bound(kp, KP_BOUNDS),
            "ti_scale_on_bound": on_bound(ti_scale, TI_SCALE_BOUNDS),
            "plant_auto_ti_s": physical_auto_ti,
            "search": "hierarchical_grid_search on the twin (paper budget)",
            "evaluations": n_evals,
        },
        "eval_model": EVAL_MODEL_KEY,
        "S_on_twin": S_twin,
        "S_achieved_on_physical": S_achieved,
        "seconds": elapsed,
        "produced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_cell("twin", key, payload)
    log(
        f"{label}: MARE={result['MARE_theta_pct']:.4f}% "
        f"Kp*={kp:.2f} TI={ti_seconds:.3f}s S_achieved={S_achieved:.4f} "
        f"({elapsed:.0f}s)"
    )
    return payload


def _paper_record_s(excitation: str) -> float:
    from backend.pipeline.payloads import paper_record_s

    value = paper_record_s(excitation, PCT_V)
    if value is None:
        raise ValueError(f"no published record length for excitation {excitation!r}")
    return value


def write_manifest(plants: list[str]) -> Path:
    manifest = {
        "atlas_version": ATLAS_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "axes": {
            "plants": plants,
            "T_log_ms": list(TLOG_MS),
            "excitations": list(EXCITATIONS),
            "LPF_hz": [lpf_key(v) for v in LPF_AXIS_HZ],
        },
        "record_s_by_excitation": {e: _paper_record_s(e) for e in EXCITATIONS},
        "fixed": {
            "pct_T": PCT_T,
            "pct_v": PCT_V,
            "Kp_star_sysid": KP_STAR_SYSID,
            "seed": SEED,
            "drift": "none",
        },
        "eval_model": EVAL_MODEL_KEY,
        "searches": {
            "reference_full": "hierarchical_grid_search, paper budget (2805 evaluations)",
            "reference_coarse": f"{COARSE_SIDE}x{COARSE_SIDE} log grid ({COARSE_SIDE**2} evaluations)",
            "twin": "hierarchical_grid_search, paper budget (2805 evaluations)",
        },
        "note": (
            "The cost S is Eq. (12) measured on the paper's STEP-RESPONSE "
            "episode (eval_model above): three +20 % steps, one channel per 5 s "
            "episode, each held, edges at 1/6/11 s, clean evaluation loop, "
            "metrics from true tensions. Main text 2.5 calls these "
            "'step-response metrics' and normalises OS and t_s per channel by "
            "that channel's own |dT_ref,i|; the 16 s E_Toggle record is the "
            "*identification* protocol, not this episode. See "
            "docs/paper-v5.1-vs-dashboard-2026-09-07.md. "
            "Twin and reference are searched at the same paper budget, so "
            "S_achieved/S_ref_full is the comparable ratio; "
            "S_achieved/S_ref_coarse is kept as a second reading. "
            "A protocol outside these axes has no cell and must read "
            "'not precomputed' -- never interpolate between neighbouring cells. "
            "Every cell records ti_scale_on_bound and kp_on_bound, so a "
            "boundary-pinned optimum is never presented as a free one; the "
            "plant's own auto_Ti is carried alongside as the engineering value."
        ),
        "reference_cells": len(plants),
        "twin_cells": len(plants) * len(TLOG_MS) * len(EXCITATIONS),
    }
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    path = ATLAS_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plants", nargs="*", default=None, help="plant ids (default: all ten)")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-twins", action="store_true")
    parser.add_argument("--force", action="store_true", help="recompute cells that already exist")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--excitations", nargs="*", default=None,
                        help="restrict the twin sweep to these excitations (default: all six)")
    args = parser.parse_args()

    from backend.validation.plants import plant_registry

    _install_eval_model()

    plants = args.plants or [str(row["plant_id"]) for row in plant_registry()]
    n_ref = len(plants)
    excitations = tuple(args.excitations) if args.excitations else EXCITATIONS
    unknown = set(excitations) - set(EXCITATIONS)
    if unknown:
        parser.error(f"unknown excitations: {sorted(unknown)}")
    n_twin = len(plants) * len(TLOG_MS) * len(excitations) * len(LPF_AXIS_HZ)
    per_eval = 0.0013  # measured on the JAX kernel, this CPU
    est_h = (n_ref * (2805 + COARSE_SIDE**2) * per_eval
             + n_twin * (2805 * per_eval + 4.0)) / 3600

    log(f"gain atlas {ATLAS_VERSION}")
    log(f"  eval model    : {EVAL_MODEL_KEY} (step-response episode, paper 2.5)")
    log(f"  plants        : {len(plants)} -> {', '.join(plants)}")
    log(f"  T_log values  : {len(TLOG_MS)}")
    log(f"  excitations   : {len(EXCITATIONS)}")
    log(f"  LPF cutoffs   : {len(LPF_AXIS_HZ)} -> {', '.join(lpf_key(v) for v in LPF_AXIS_HZ)}")
    log(f"  reference     : {n_ref} cells (full 2805 + {COARSE_SIDE**2} coarse each)")
    log(f"  twin          : {n_twin} cells (full HGS, 2805 evaluations each)")
    log(f"  estimate      : {est_h:.1f} h")
    log(f"  output        : {CELL_DIR}")

    if args.dry_run:
        log("dry run, nothing computed")
        return 0

    write_manifest(plants)
    started = time.time()

    if not args.skip_reference:
        log("--- reference searches on theta_true ---")
        for i, plant_id in enumerate(plants, 1):
            log(f"[{i}/{len(plants)}] reference {plant_id}")
            build_reference(plant_id, args.force)

    if not args.skip_twins:
        log("--- twin searches per protocol ---")
        done = 0
        for plant_id in plants:
            for tlog in TLOG_MS:
                for excitation in excitations:
                    for lpf in LPF_AXIS_HZ:
                        done += 1
                        out = build_twin_cell(plant_id, tlog, excitation, lpf, args.force)
                        if out is None:
                            continue
                        elapsed = time.time() - started
                        rate = elapsed / done
                        remaining = (n_twin - done) * rate / 3600
                        log(f"    [{done}/{n_twin}] ~{remaining:.1f} h remaining")

    log(f"atlas complete in {(time.time() - started)/3600:.2f} h")
    write_manifest(plants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
