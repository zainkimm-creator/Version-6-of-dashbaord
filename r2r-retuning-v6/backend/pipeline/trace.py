"""Acquisition playback: the identification record the physical plant runs.

The machine screen animates with REAL data, not a cartoon. This runs the
protocol's excitation on the (drifted) physical plant at the SysID-mode gain --
the same record `retuning.identify_twin` logs, with the same noise, filter and
seeds -- and returns a decimated time series of tensions, set-points and roller
speeds. When theta-hat is supplied, the twin rebuilt from it runs the same
record so the screen can show plant and twin side by side.

A multi-record excitation (ET3M) returns its first record and says so.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from backend.models.simulation import SimulationConfig, simulate
from backend.validation.paper_inputs import build_excitation, excitation_records
from backend.validation.retuning import RETUNING_CAMPAIGN_GROUP, STEP_FRACTION

from . import cache
from .payloads import RunSpec, run_hash
from .plant import drifted_params, sysid_controller
from .retune import twin_params_from_estimates

TRACE_VERSION = "v1"
CACHE_KIND = f"trace-{TRACE_VERSION}"
MAX_POINTS = 600


def _round(rows, digits: int = 4) -> list[list[float]]:
    return [[round(float(v), digits) for v in row] for row in rows]


def _run_record(params, run: RunSpec, meta: Mapping[str, Any]) -> tuple[list[dict[str, float]], dict[str, Any]]:
    protocol = run.protocol
    records = excitation_records(protocol.excitation, RETUNING_CAMPAIGN_GROUP)
    schedule = records[0]
    speed = float(meta["v_ref_m_s"]) * float(schedule.v_ref_multiplier)
    duration = float(protocol.record_s) if len(records) == 1 else float(schedule.duration_s)
    config = SimulationConfig(
        duration_s=duration,
        dt_s=0.001,
        controller_sample_time_s=0.001,
        log_sample_time_s=max(0.001, float(protocol.T_log_ms) / 1000.0),
        line_speed_m_s=speed,
        sensor_noise_tension_N=float(protocol.pct_T) * float(meta["T_max_N"]),
        sensor_noise_velocity_m_s=float(protocol.pct_v) * float(meta["v_max_m_s"]),
        sensor_lpf_hz=protocol.LPF_T_hz,
        velocity_lpf_hz=protocol.LPF_v_hz,
        noise_rng="numpy_default_rng",
        velocity_seed_offset=100,
        seed=int(protocol.seed),
        controller_tracks_drift=False,
    )
    step_n = STEP_FRACTION * float(params.tension_ref_N[0])
    result = simulate(
        params=params,
        controller_config=sysid_controller(params, speed, float(protocol.Kp_star)),
        config=config,
        excitation=build_excitation(schedule, step_n),
        write_output=False,
    )
    info = {"records_total": len(records), "record_s": duration, "line_speed_m_s": speed,
            "step_N": step_n}
    return result.rows, info


def acquisition_trace(
    run: RunSpec,
    estimates: Mapping[str, float] | None = None,
    *,
    use_cache: bool = True,
    max_points: int = MAX_POINTS,
) -> dict[str, Any]:
    """Plant (and twin) tensions over the protocol's identification record."""

    blob = json.dumps({"run": run_hash(run), "v": TRACE_VERSION, "n": max_points,
                       "est": sorted((k, round(float(v), 12)) for k, v in (estimates or {}).items())})
    key = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    if use_cache:
        hit = cache.load(CACHE_KIND, key)
        if hit is not None:
            return {**hit, "cached": True}

    base, drifted, meta = drifted_params(run.plant, run.drift)
    rows, info = _run_record(drifted, run, meta)
    stride = max(1, math.ceil(len(rows) / max_points))
    picked = rows[::stride]

    def series(source, names):
        return _round([[r[name] for name in names] for r in source])

    twin_tensions = None
    if estimates:
        twin = twin_params_from_estimates(drifted, estimates)
        twin_rows, _ = _run_record(twin, run, meta)
        twin_tensions = series(twin_rows[::stride], ("T1", "T2", "T3"))

    tensions = series(picked, ("T1", "T2", "T3"))
    if not all(math.isfinite(v) for row in tensions for v in row):
        raise ValueError("the plant diverged under this protocol; nothing to play back")
    times = [round(float(r["time_s"]), 4) for r in picked]
    payload = {
        "trace_version": TRACE_VERSION,
        "run_hash": run_hash(run),
        "excitation": run.protocol.excitation,
        "records_total": info["records_total"],
        "note": ("first of the excitation's records shown" if info["records_total"] > 1 else None),
        "record_s": info["record_s"],
        "dt_s": (times[1] - times[0]) if len(times) > 1 else info["record_s"],
        "line_speed_m_s": info["line_speed_m_s"],
        "step_N": info["step_N"],
        "T_ref_base_N": [float(v) for v in drifted.tension_ref_N],
        "R_m": [float(v) for v in drifted.roller_radius_m],
        "sigma_T_N": float(run.protocol.pct_T) * float(meta["T_max_N"]),
        "t_s": times,
        "T_N": tensions,
        "T_ref_N": series(picked, ("T1_ref_N", "T2_ref_N", "T3_ref_N")),
        "v_m_s": series(picked, ("v_UW_m_s", "v_Nip_m_s", "v_RW_m_s")),
        "omega_rad_s": series(picked, ("omega_UW", "omega_Nip", "omega_RW")),
        "twin_T_N": twin_tensions,
        "samples": len(times),
    }
    cache.store(CACHE_KIND, key, payload)
    return {**payload, "cached": False}
