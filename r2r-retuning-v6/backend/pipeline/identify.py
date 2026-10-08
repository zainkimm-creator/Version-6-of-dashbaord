"""Identification: a RunSpec becomes the two-column theta_true / theta_hat panel.

This is the hinge of the whole dashboard. It runs the excitation the protocol
names on the (drifted) physical twin, logs at the protocol's rate under the
protocol's noise, and fits the v5 operating-point-weighted PEM.

Non-convergence is reported as a failure with a null MARE, never as a large
number -- a diverged fit is not a bad estimate, it is no estimate.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from backend.models.equations import PARAMETER_NAMES
from backend.validation.retuning import Protocol, identify_twin

from . import cache
from .payloads import ProtocolSpec, RunSpec, run_hash
from .plant import drifted_params

# Bump this whenever the way an identification is COMPUTED changes.
#
# The cache is content-addressed on `run_hash(RunSpec)` -- the plant, the drift
# and the protocol. That addresses the INPUTS, and says nothing about the code
# that turned them into a result, so a change to the estimator or the
# acquisition silently keeps serving results computed the old way. Folding a
# version into the cache namespace makes a computation change a cache miss,
# and leaves the old entries on disk under their own namespace rather than
# deleting work that may still be wanted for comparison.
#
# v2 (2026-09-11): a multi-record excitation is now simulated once per record,
# at that record's own line speed and seed, and all records enter ONE joint fit
# through the multi-condition cost of Eq. (11). Before this, ET3M ran a single
# record at a single operating point, so every cached ET3M entry answers for a
# different excitation.
IDENTIFY_VERSION = "v2"
CACHE_KIND = f"identify-{IDENTIFY_VERSION}"


def protocol_to_paper_protocol(spec: ProtocolSpec) -> Protocol:
    """Map the session protocol onto the campaign's `Protocol` dataclass."""

    return Protocol(
        name="session",
        log_sample_time_s=spec.T_log_ms / 1000.0,
        tension_noise_fraction=spec.pct_T,
        velocity_noise_fraction=spec.pct_v,
        tension_lpf_hz=spec.LPF_T_hz,
        velocity_lpf_hz=spec.LPF_v_hz,
        record_duration_s=spec.record_s,
    )


def _rows(theta_true: dict[str, float], estimates: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in PARAMETER_NAMES:
        truth = float(theta_true[name])
        estimate = float(estimates[name])
        rows.append(
            {
                "parameter": name,
                "theta_true": truth,
                "theta_hat": estimate,
                "error_pct": 100.0 * (estimate - truth) / truth if truth else None,
            }
        )
    return rows


def identify(run: RunSpec, *, use_cache: bool = True) -> dict[str, Any]:
    """Run the protocol on the physical twin and fit theta-hat."""

    key = run_hash(run)
    if use_cache:
        hit = cache.load(CACHE_KIND, key)
        if hit is not None:
            return {**hit, "cached": True}

    base, drifted, meta = drifted_params(run.plant, run.drift)
    plant_meta = {
        "v0_mps": meta["v_ref_m_s"],
        "T_max_N": meta["T_max_N"],
        "v_max_mps": meta["v_max_m_s"],
    }

    _, info = identify_twin(
        drifted,
        protocol_to_paper_protocol(run.protocol),
        plant_meta,
        seed=run.protocol.seed,
        kp_star=run.protocol.Kp_star,
        excitation=run.protocol.excitation,
    )

    converged = bool(info["converged"])
    theta_true = {
        name: float(value) for name, value in drifted.sysid_values().items()
    }
    payload: dict[str, Any] = {
        "run_hash": key,
        "plant_id": meta["plant_id"],
        "rows": _rows(theta_true, info["estimates"]) if converged else [],
        "estimates": info["estimates"] if converged else None,
        "MARE_theta_pct": float(info["mare_theta_percent"]) if converged else None,
        "converged": converged,
        "optimizer_status": info["optimizer_status"],
        "nfev": info["nfev"],
        "max_nfev": info["max_nfev"],
        "effective_record_s": float(run.protocol.record_s),
        "n_samples": int(info.get("samples") or
                         run.protocol.record_s / (run.protocol.T_log_ms / 1000.0)),
        "excitation": info["excitation"],
        # How the excitation was actually acquired and fitted. ET3M logs three
        # records at v0 x {0.5, 1, 2} with seeds base + 17*i and identifies them
        # in ONE joint fit (paper Eq. 11); every other type logs one record.
        "records": info.get("records"),
        "operating_point_multipliers": info.get("operating_point_multipliers"),
        "record_seeds": info.get("record_seeds"),
        "fit": info.get("fit"),
        "Kp_star": info["kp_star"],
        "produced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    cache.store(CACHE_KIND, key, payload)
    return {**payload, "cached": False}
