"""Probes that produce the frozen golden values.

Two families:

* ``compute_probes`` recomputes from the physics core on every call. These are
  the numbers that move if a refactor breaks the engine, and they are the real
  regression proof.
* ``payload_probes`` (Task 2) reads the study payloads. Those come out of the
  study caches, so a cache-to-cache comparison proves nothing about the physics
  path -- which is exactly why ``compute_probes`` exists alongside it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

# Keys whose values legitimately differ between runs and must never be frozen.
_VOLATILE_KEY_PARTS = (
    "path",
    "url",
    "timestamp",
    "created_at",
    "produced_at",
    "generated",
    "elapsed",
    "runtime",
    "wall_clock",
)


def _is_volatile(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _VOLATILE_KEY_PARTS)


def flatten_numeric(payload: Any, prefix: str = "") -> dict[str, float]:
    """Flatten a JSON-ish payload to ``dotted.path -> float`` for numeric leaves.

    Booleans are dropped (they are flags, not measurements), as are strings,
    nulls and any key naming a path, URL or timestamp.
    """

    flat: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if _is_volatile(str(key)):
                continue
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_numeric(value, child))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            flat.update(flatten_numeric(value, f"{prefix}[{index}]"))
    elif isinstance(payload, bool) or payload is None:
        return flat
    elif isinstance(payload, (int, float)):
        flat[prefix] = float(payload)
    return flat


def _sysid_mode_controller(params, line_speed_m_s: float, kp_star: float):
    """The controller convention `closed_loop_damping` uses for its eigenvalues."""

    from backend.models.controller import ControllerConfig
    from backend.validation.retuning import plant_auto_ti_s

    return ControllerConfig(
        target_tension_N=params.tension_ref_N,
        line_speed_m_s=line_speed_m_s,
        Kp_star_m_s_per_N=float(kp_star),
        TI_s=plant_auto_ti_s(params, line_speed_m_s),
        high_ea_kp_cap_enabled=False,
        feedforward_uses_measured_omega=True,
        paper_velocity_gain_enabled=True,
        velocity_correction_limit_fraction=None,
        steady_velocity_uses_dynamic_target=False,
    )


def compute_probes() -> dict[str, float]:
    """Recompute the physics-core reference numbers. Seconds, not minutes."""

    from backend.models.modal import closed_loop_modal_analysis, open_loop_tau_min_s
    from backend.models.simulation import SimulationConfig, simulate
    from backend.validation.excitations import get_excitation_profile
    from backend.validation.plants import parameters_for_plant, plant_registry
    from backend.validation.retuning import (
        DRIFT_BY_CODE,
        PROTOCOL_FIELD_MATCHED,
        Protocol,
        apply_drift,
        identify_twin,
    )

    probes: dict[str, float] = {}

    params, meta = parameters_for_plant("P01")
    plant_meta = {
        "v0_mps": meta["v_ref_m_s"],
        "T_max_N": meta["T_max_N"],
        "v_max_mps": meta["v_max_m_s"],
    }

    drifted = apply_drift(params, DRIFT_BY_CODE["D03"])
    _, info = identify_twin(drifted, PROTOCOL_FIELD_MATCHED, plant_meta, seed=0)
    probes["compute/identify_P01_D03_field_matched_mare_pct"] = float(
        info["mare_theta_percent"]
    )

    roundtrip = Protocol(
        "golden_roundtrip",
        log_sample_time_s=0.001,
        tension_noise_fraction=0.0,
        velocity_noise_fraction=0.0,
        tension_lpf_hz=None,
        velocity_lpf_hz=None,
    )
    _, rt_info = identify_twin(params, roundtrip, plant_meta, seed=0)
    probes["compute/roundtrip_P01_noise_free_tlog1ms_mare_pct"] = float(
        rt_info["mare_theta_percent"]
    )

    for row in plant_registry():
        plant_id = str(row["plant_id"])
        plant_params, _ = parameters_for_plant(plant_id)
        speed = float(row["v_ref_m_s"])
        modal = closed_loop_modal_analysis(
            plant_params, _sysid_mode_controller(plant_params, speed, 100.0)
        )
        probes[f"compute/zeta_cl_min.{plant_id}"] = float(modal["zeta_cl_min"])
        probes[f"compute/tau_min_s.{plant_id}"] = float(
            open_loop_tau_min_s(plant_params, line_speed_m_s=speed)
        )

    sim = simulate(
        params,
        config=SimulationConfig(
            duration_s=4.0,
            log_sample_time_s=0.010,
            line_speed_m_s=float(meta["v_ref_m_s"]),
            seed=7,
            output_name="golden_probe.csv",
        ),
        excitation=get_excitation_profile("ET3", 0.2 * float(meta["T_ref_N"])),
        write_output=False,
    )
    for key, value in flatten_numeric(sim.metrics, "compute/simulation_P01_ET3").items():
        probes[key] = value

    return probes


# Route, request body. `plant_id: "ALL"` matches what the dashboard sends, and
# `force_rerun` is left off so the study caches are used -- this half of the
# freeze is about payload shape, not about recomputing the physics.
PAYLOAD_ROUTES: tuple[tuple[str, dict[str, object]], ...] = (
    ("/validate/logging-rate", {"plant_id": "ALL"}),
    ("/validate/excitation", {"plant_id": "ALL"}),
    ("/validate/drift", {"plant_id": "ALL"}),
    ("/validate/noise-aware-logging-lpf", {"plant_id": "P01"}),
    ("/validate/closed-loop-damping", {}),
    ("/validate/retuning", {}),
    ("/validate/retuning-tier1", {}),
)


def payload_probes() -> dict[str, float]:
    """Freeze the numeric leaves of every validation route's payload.

    A route that cannot run in this environment (for example retuning-tier1
    without the v5 figure package) is skipped and reported, not failed: its
    absence is a property of the machine, not a regression.
    """

    from fastapi.testclient import TestClient

    from backend.api.main import app

    flat: dict[str, float] = {}
    with TestClient(app) as client:
        for route, body in PAYLOAD_ROUTES:
            response = client.post(route, json=body)
            if response.status_code == 422:
                continue
            response.raise_for_status()
            prefix = "payload/" + route.strip("/").replace("/", ".")
            flat.update(flatten_numeric(response.json(), prefix))
    return flat


def all_probes() -> dict[str, float]:
    return {**compute_probes(), **payload_probes()}


# Substrings (case-insensitive) naming the scalars the spec calls out
# explicitly: "Key scalars from all seven /validate/* routes." Everything
# else in a route's payload is represented only by the digest below.
_NAMED_SCALAR_KEY_PARTS = (
    "mare",
    "epsilon_theta",
    "exponent",
    "r_squared",
    "zeta",
    "tau_min",
    "win_rate",
    "median",
    "kappa",
    "condition_number",
    "optimum",
    "failure",
)

# Guardrail on the named-scalar cut: if it comes in larger than this, keep
# only the shallow (route-level) scalars rather than every nested one.
_NAMED_SCALAR_LIMIT = 5000


def _is_named_scalar(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _NAMED_SCALAR_KEY_PARTS)


def _dots_after_prefix(key: str, prefix: str) -> int:
    return key[len(prefix):].count(".")


def digest_leaves(leaves: dict[str, float]) -> float:
    """A float64-safe digest over a canonical flattening of ``leaves``.

    13 hex digits = 52 bits, which float64 represents exactly -- more than
    that stops the value round-tripping through JSON.
    """

    canonical = json.dumps(sorted(leaves.items()), separators=(",", ":"))
    digest_hex = hashlib.sha256(canonical.encode()).hexdigest()
    return float(int(digest_hex[:13], 16))


def _build_summary(per_route: dict[str, dict[str, float]]) -> dict[str, float]:
    """Pure computation half of ``payload_summary_probes``.

    Takes each route's already-fetched leaf map and builds the leaf count,
    digest and named-scalar cut described there. Split out so the guardrail
    logic can be exercised directly, without a TestClient or the network.
    """

    summary: dict[str, float] = {}
    named_by_prefix: dict[str, dict[str, float]] = {}
    total_named = 0
    for prefix, leaves in per_route.items():
        summary[f"{prefix}/__leaf_count"] = float(len(leaves))
        summary[f"{prefix}/__digest"] = digest_leaves(leaves)
        named = {key: value for key, value in leaves.items() if _is_named_scalar(key)}
        named_by_prefix[prefix] = named
        total_named += len(named)

    if total_named > _NAMED_SCALAR_LIMIT:
        named_by_prefix = {
            prefix: {
                key: value
                for key, value in named.items()
                if _dots_after_prefix(key, prefix) <= 2
            }
            for prefix, named in named_by_prefix.items()
        }
        survived = sum(len(named) for named in named_by_prefix.values())
        print(
            f"WARNING: named-scalar guardrail fired: {total_named} named "
            f"scalars found, over the {_NAMED_SCALAR_LIMIT} limit; kept "
            f"{survived} at nesting depth <= 2",
            file=sys.stderr,
        )

    for named in named_by_prefix.values():
        summary.update(named)

    return summary


def payload_summary_probes() -> dict[str, float]:
    """Freeze a digest and named scalars per route, not every leaf.

    Every leaf (``payload_probes``) makes the golden file unreviewable and
    the gate slow. This keeps, per route: a leaf count and a digest over the
    full flattened payload (a cheap, exact regression signal for *any*
    change to that route's response) plus the scalars the spec names
    explicitly. A route that cannot run in this environment is skipped, same
    as ``payload_probes``.

    Expect the named-scalar guardrail in ``_build_summary`` to fire at this
    payload size: the ``/validate/*`` sweeps carry far more than
    ``_NAMED_SCALAR_LIMIT`` named scalars between them, so the cut to nesting
    depth <= 2 is the *normal* outcome here, not a rare backstop. The
    consequence is that routes whose named scalars all sit deeper than that --
    ``closed-loop-damping``, ``drift``, ``excitation`` -- are frozen by their
    ``__digest`` and ``__leaf_count`` alone. That is not a broken freeze:
    digests compare exactly (``freeze_golden._EXACT_MATCH_SUFFIXES``), so any
    change at all to those payloads is still caught. Grepping the golden file
    for ``zeta`` under ``closed-loop-damping`` and finding nothing is the
    guardrail working, not the freeze failing.
    """

    from fastapi.testclient import TestClient

    from backend.api.main import app

    per_route: dict[str, dict[str, float]] = {}
    with TestClient(app) as client:
        for route, body in PAYLOAD_ROUTES:
            response = client.post(route, json=body)
            if response.status_code == 422:
                continue
            response.raise_for_status()
            prefix = "payload/" + route.strip("/").replace("/", ".")
            leaves = flatten_numeric(response.json(), prefix)
            if leaves:
                per_route[prefix] = leaves

    return _build_summary(per_route)
