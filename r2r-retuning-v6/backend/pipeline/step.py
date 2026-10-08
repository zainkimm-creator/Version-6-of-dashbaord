"""The cost test's step response, for the screen's two step-response windows.

The retuning cost is scored on one experiment (paper §4.2, authors' reply):
run the line at its set-points, step all three tension references +20 % at
t = 5 s, simulate to 30 s. `retune._validation_response` runs exactly that to
compute the twin-validation RMSE but keeps only the post-step tensions. This
module runs the same experiment and returns the whole decimated trace, plus the
per-zone step metrics a commissioning engineer reads off it (overshoot, time
constant, settling time), for either the drifted physical plant or the twin
rebuilt from an identification. Same gains on both is the point: the gap
between the two windows is the twin error, seen as a response.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from backend.models.controller import ControllerConfig
from backend.models.simulation import SimulationConfig, simulate

from . import cache
from .payloads import RunSpec, run_hash
from .plant import drifted_params
from .retune import STEP_FRACTION, T_SIM_S, T_STEP_S, twin_params_from_estimates

# v3 (7 Oct 2026): settling_time_s is now the TOTAL time outside the +-2 % band,
# not the last sample outside it. Without this bump the cache serves the old value.
STEP_VERSION = "v3"
CACHE_KIND = f"step-{STEP_VERSION}"
MAX_POINTS = 600
ZONES = ("UW", "OutFeeder", "RW")
# The settling band and the "stable" verdict share the paper's +-2 % of the step.
SETTLE_BAND = 0.02


def _three(value: float | Sequence[float], name: str) -> tuple[float, float, float]:
    values = (value, value, value) if isinstance(value, (int, float)) else tuple(value)
    if len(values) != 3 or not all(math.isfinite(float(v)) and float(v) > 0 for v in values):
        raise ValueError(f"{name} must be a positive scalar or three positive values")
    return tuple(float(v) for v in values)  # type: ignore[return-value]


def _first_time(times: np.ndarray, mask: np.ndarray) -> float | None:
    hits = np.flatnonzero(mask)
    return float(times[hits[0]]) if hits.size else None


def _zone_metrics(times: np.ndarray, tension: np.ndarray, base: float) -> dict[str, float | None]:
    """Step metrics on one span, times measured from the step instant."""

    step = STEP_FRACTION * base
    final = base + step
    post = times > T_STEP_S
    t = times[post] - T_STEP_S
    y = tension[post]
    peak_i = int(np.argmax(y))
    peak = float(y[peak_i])
    inside = np.abs(y - final) <= SETTLE_BAND * step
    # Settling time: the TOTAL time outside the +-2 % band -- the same quantity the
    # cost scores. The previous definition (last sample outside the band) jumps by a
    # whole oscillation when a wobble grazes the band, which is where the chart's
    # anomalies came from: +72 % on a 1.14 % gain change, settling 1.06 -> 1.46 s.
    outside_after = np.flatnonzero(~inside)
    dt_s = float(t[1] - t[0]) if t.size > 1 else 0.0
    settle = float(outside_after.size) * dt_s
    # kept for reference, under its own name
    last_out = float(t[outside_after[-1] + 1]) if outside_after.size and outside_after[-1] + 1 < t.size else (
        0.0 if not outside_after.size else None)
    return {
        "T_ref_N": base,
        "T_final_N": final,
        "step_N": step,
        "peak_N": peak,
        "peak_time_s": float(t[peak_i]),
        "overshoot_pct": max(0.0, (peak - final) / step * 100.0),
        "time_constant_s": _first_time(t, y >= base + 0.632 * step),
        "t90_s": _first_time(t, y >= base + 0.9 * step),
        "settling_time_s": settle,
        "settling_last_outside_s": last_out,
        "time_outside_band_s": settle,
        "rmse_after_step_N": float(np.sqrt(np.mean((y - final) ** 2))),
    }


def _run(params, line_speed_m_s: float, kp: tuple[float, float, float], ti: tuple[float, float, float],
         *, noise_sigma_N: float = 0.0, lpf_hz: float | None = None, seed: int = 0):
    """The cost test, exactly as the cost scores it (no noise in the loop, so
    the metrics are the cost's own numbers). With `noise_sigma_N` > 0 a
    measured trace -- the true tension plus the sensor's noise, through the
    anti-alias filter -- is returned beside it: the trace a line's HMI would
    actually draw, which never sits perfectly still inside the band."""

    base = np.asarray(params.tension_ref_N, dtype=float)
    step = STEP_FRACTION * base

    def profile(t_s: float) -> tuple[float, float, float]:
        return tuple(float(v) for v in step) if t_s >= T_STEP_S else (0.0, 0.0, 0.0)  # type: ignore[return-value]

    config = SimulationConfig(
        duration_s=T_SIM_S, dt_s=0.001, controller_sample_time_s=0.001, log_sample_time_s=0.001,
        line_speed_m_s=line_speed_m_s, controller_tracks_drift=False, sensor_lpf_hz=None,
    )
    controller = ControllerConfig(
        target_tension_N=tuple(base), line_speed_m_s=line_speed_m_s,
        Kp_star_m_s_per_N=kp, TI_s=ti,
        paper_velocity_gain_enabled=True, high_ea_kp_cap_enabled=False,
        velocity_correction_limit_fraction=None,
    )
    result = simulate(params=params, controller_config=controller, config=config,
                      excitation=profile, write_output=False)
    times = np.array([float(r["time_s"]) for r in result.rows])
    tensions = np.array([[float(r["T1"]), float(r["T2"]), float(r["T3"])] for r in result.rows])
    measured = None
    if noise_sigma_N > 0:
        rng = np.random.default_rng(int(seed) + 7)
        raw = tensions + rng.normal(0.0, float(noise_sigma_N), tensions.shape)
        if lpf_hz:
            alpha = 2 * math.pi * float(lpf_hz) * 0.001 / (1 + 2 * math.pi * float(lpf_hz) * 0.001)
            measured = np.empty_like(raw); measured[0] = raw[0]
            for k in range(1, raw.shape[0]):
                measured[k] = alpha * raw[k] + (1 - alpha) * measured[k - 1]
        else:
            measured = raw
    return times, tensions, measured


def step_response(
    run: RunSpec,
    kp_star: float | Sequence[float],
    ti_s: float | Sequence[float],
    estimates: Mapping[str, float] | None = None,
    *,
    measured: bool = False,
    seed: int = 0,
    use_cache: bool = True,
    max_points: int = MAX_POINTS,
) -> dict[str, Any]:
    """The +20 % cost-test step on the drifted plant, or on the twin from `estimates`.

    `measured` adds the trace the line would see: the true tension plus the
    protocol's sensor noise through its anti-alias filter. The response and
    the metrics are the cost's own (noise-free), so they agree with S.
    """

    kp = _three(kp_star, "kp_star")
    ti = _three(ti_s, "ti_s")
    blob = json.dumps({"run": run_hash(run), "v": STEP_VERSION, "n": max_points, "kp": kp, "ti": ti,
                       "measured": bool(measured), "seed": int(seed),
                       "est": sorted((k, round(float(v), 12)) for k, v in (estimates or {}).items())})
    key = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    if use_cache:
        hit = cache.load(CACHE_KIND, key)
        if hit is not None:
            return {**hit, "cached": True}

    _, drifted, meta = drifted_params(run.plant, run.drift)
    subject = drifted if not estimates else twin_params_from_estimates(drifted, estimates)
    v0 = float(meta["v_ref_m_s"])
    sigma = float(run.protocol.pct_T) * float(meta["T_max_N"]) if measured else 0.0
    times, tensions, meas = _run(subject, v0, kp, ti, noise_sigma_N=sigma,
                                 lpf_hz=run.protocol.LPF_T_hz, seed=seed)
    if not np.all(np.isfinite(tensions)):
        raise ValueError("the step test diverged with these gains; nothing to show")
    stride = max(1, math.ceil(times.size / max_points))
    payload = {
        "step_version": STEP_VERSION,
        "run_hash": run_hash(run),
        "plant_id": meta["plant_id"],
        "subject": "twin" if estimates else "plant",
        "kp_star_per_zone": list(kp),
        "ti_s_per_zone": list(ti),
        "zones": list(ZONES),
        "step_time_s": T_STEP_S,
        "duration_s": T_SIM_S,
        "T_ref_base_N": [float(v) for v in subject.tension_ref_N],
        "T_ref_final_N": [float(v) * (1.0 + STEP_FRACTION) for v in subject.tension_ref_N],
        "t_s": [round(float(v), 4) for v in times[::stride]],
        "T_N": [[round(float(v), 4) for v in row] for row in tensions[::stride]],
        "T_meas_N": ([[round(float(v), 4) for v in row] for row in meas[::stride]] if meas is not None else None),
        "measured": bool(measured),
        "noise_sigma_N": sigma,
        "lpf_hz": run.protocol.LPF_T_hz if measured else None,
        "metrics": [_zone_metrics(times, tensions[:, i], float(subject.tension_ref_N[i])) for i in range(3)],
        "samples": int(times[::stride].size),
    }
    cache.store(CACHE_KIND, key, payload)
    return {**payload, "cached": False}


# --------------------------------------------------------------------------- #
# the line as it runs today: the commissioned gains on the drifted plant
# --------------------------------------------------------------------------- #
BASELINE_KIND = f"baseline-{STEP_VERSION}"


def baseline(run: RunSpec, cost_tier: str | None = None, *, measured: bool = False, seed: int = 0,
             use_cache: bool = True) -> dict[str, Any]:
    """RUN PHYSICAL MACHINE: the gains the line was commissioned with, and the
    cost test they give on the plant as it has drifted.

    The commissioned pair is Algorithm 1's starting point -- the paper's shared
    (K_p*, T_I) found on the pre-drift plant -- the same search `retune` runs
    for its C_target, so the two agree exactly. It is what a line keeps running
    after the rolls, bearings and web have changed, which is the whole reason
    to retune.
    """

    from backend.validation import retuning_paper_protocol as AP

    from .retune import DEFAULT_COST_TIER, _Scorer, _gains_block

    tier = cost_tier or DEFAULT_COST_TIER
    blob = json.dumps({"run": run_hash(run), "v": STEP_VERSION, "tier": tier, "measured": bool(measured), "seed": int(seed)})
    key = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    if use_cache:
        hit = cache.load(BASELINE_KIND, key)
        if hit is not None:
            return {**hit, "cached": True}

    base, drifted, meta = drifted_params(run.plant, run.drift)
    v0 = float(meta["v_ref_m_s"])
    commissioning = AP.hierarchical_grid_search(_Scorer(base, v0, tier).batch)
    gains = _gains_block(commissioning.kp, commissioning.ti)
    on_plant = _Scorer(drifted, v0, tier).breakdown(commissioning.kp, commissioning.ti)
    on_base = _Scorer(base, v0, tier).breakdown(commissioning.kp, commissioning.ti)
    payload = {
        "step_version": STEP_VERSION,
        "run_hash": run_hash(run),
        "plant_id": meta["plant_id"],
        "cost_tier": tier,
        "gains": {**gains, "on_plant": on_plant, "at_commissioning": on_base},
        "drift_applied": not run.drift.is_identity,
        "step": step_response(run, gains["kp_star_per_zone"], gains["ti_s_per_zone"],
                              measured=measured, seed=seed, use_cache=use_cache),
    }
    cache.store(BASELINE_KIND, key, payload)
    return {**payload, "cached": False}
