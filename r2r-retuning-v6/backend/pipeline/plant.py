"""Plant construction and the derived readouts the Plant tab shows.

Presets go through `parameters_for_plant` unchanged, because every frozen
study number depends on that path. Custom plants build `R2RParameters`
directly and carry no paper reference -- a hand-edited plant is not in the
paper's ten-plant table, so its published cells are `not_published`, never
interpolated.

Nothing here simulates. `derive` is eigenvalues and algebra, so it returns in
milliseconds and the frontend may call it on every slider change.
"""

from __future__ import annotations

from typing import Any

from backend.models.controller import ControllerConfig
from backend.models.equations import PARAMETER_NAMES, R2RParameters
from backend.models.modal import closed_loop_modal_analysis, open_loop_tau_min_s
from backend.validation.plants import parameters_for_plant
from backend.validation.retuning import DriftScenario, apply_drift, plant_auto_ti_s

from .payloads import DriftSpec, PlantSpec, RunSpec

# The paper's full-scale convention: a channel's full scale is its set-point
# divided by 0.30.
FULL_SCALE_FRACTION = 0.30

# Section 3.1's speed split. The paper names only the two ends; anything
# between them is reported as "mid" rather than forced into one of them.
SLOW_MAX_V0 = 0.5
FAST_MIN_V0 = 2.0


def params_from_spec(spec: PlantSpec) -> tuple[R2RParameters, dict[str, Any]]:
    """Return `(params, meta)` for a preset or a hand-edited plant."""

    if spec.source == "preset":
        params, plant = parameters_for_plant(spec.preset_id)
        meta = {
            "plant_id": str(plant["plant_id"]),
            "source": "preset",
            "label": plant["label"],
            "v_ref_m_s": float(plant["v_ref_m_s"]),
            "v_max_m_s": float(plant["v_max_m_s"]),
            "T_ref_N": float(plant["T_ref_N"]),
            "T_max_N": float(plant["T_max_N"]),
            "paper_reference": {
                "zeta_cl_min": float(plant["zeta_cl_min"]),
                "source": "data/paper_reference/ten_plant_parameters.csv",
            },
        }
        return params, meta

    if spec.T_ref is None or spec.T_ref <= 0:
        raise ValueError("T_ref must be positive")

    params = R2RParameters(
        span_length_m=spec.L,
        roller_radius_m=spec.R,
        inertia_kg_m2=spec.J,
        tension_ref_N=(spec.T_ref,) * 3,
        process_noise_b=0.0,
        kf_UW=spec.f[0],
        kf_Nip=spec.f[1],
        kf_RW=spec.f[2],
        EA=spec.EA,
        feeder_velocity_m_s=spec.v0,
    )
    meta = {
        "plant_id": "CUSTOM",
        "source": "custom",
        "label": "Custom plant",
        "v_ref_m_s": float(spec.v0),
        "v_max_m_s": float(spec.v0) / FULL_SCALE_FRACTION,
        "T_ref_N": float(spec.T_ref),
        "T_max_N": float(spec.T_ref) / FULL_SCALE_FRACTION,
        "paper_reference": {"status": "not_published"},
    }
    return params, meta


def drift_scenario(drift: DriftSpec) -> DriftScenario:
    """Percentages become the absolute multipliers `apply_drift` expects."""

    return DriftScenario(
        code="LIVE",
        label="session drift",
        EA_scale=1.0 + drift.EA_pct / 100.0,
        friction_scale=1.0 + drift.f_pct / 100.0,
        J_UW_scale=1.0 + drift.J_UW_pct / 100.0,
        J_Nip_scale=1.0 + drift.J_Nip_pct / 100.0,
        J_RW_scale=1.0 + drift.J_RW_pct / 100.0,
    )


def drifted_params(
    spec: PlantSpec, drift: DriftSpec
) -> tuple[R2RParameters, R2RParameters, dict[str, Any]]:
    base, meta = params_from_spec(spec)
    if drift.is_identity:
        return base, base, meta
    return base, apply_drift(base, drift_scenario(drift)), meta


def sysid_controller(
    params: R2RParameters, line_speed_m_s: float, kp_star: float
) -> ControllerConfig:
    """The controller convention the closed-loop damping study uses."""

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


def _speed_class(v0: float) -> str:
    if v0 <= SLOW_MAX_V0:
        return "slow"
    if v0 >= FAST_MIN_V0:
        return "fast"
    return "mid"


def _theta(params: R2RParameters) -> dict[str, float]:
    values = params.sysid_values()
    return {name: float(values[name]) for name in PARAMETER_NAMES}


def derive(run: RunSpec) -> dict[str, Any]:
    """The `/plant/derive` body: everything the Plant tab shows, no simulation."""

    base, drifted, meta = drifted_params(run.plant, run.drift)
    line_speed = float(meta["v_ref_m_s"])
    kp_star = float(run.protocol.Kp_star)

    # `line_speed` is passed explicitly, the same as `open_loop_tau_min_s`
    # below. The default would read it off `drifted.feeder_velocity_m_s`,
    # which happens to equal `meta["v_ref_m_s"]` only because `apply_drift`
    # does not scale the feeder velocity -- an invariant this module should
    # not be relying on silently.
    modal = closed_loop_modal_analysis(
        drifted,
        sysid_controller(drifted, line_speed, kp_star),
        line_speed_m_s=line_speed,
    )
    t_max = float(meta["T_max_N"])
    v_max = float(meta["v_max_m_s"])

    return {
        "plant_id": meta["plant_id"],
        "source": meta["source"],
        "label": meta["label"],
        "theta_true": _theta(base),
        "theta_drifted": _theta(drifted),
        "drift_applied": not run.drift.is_identity,
        "T_ref_N": float(meta["T_ref_N"]),
        "T_max_N": t_max,
        "v0_m_s": line_speed,
        "v_max_m_s": v_max,
        "T_I_s": float(plant_auto_ti_s(drifted, line_speed)),
        "Kp_star": kp_star,
        "zeta_cl_min": float(modal["zeta_cl_min"]),
        "regime": str(modal["regime"]),
        "stable": bool(modal["stable"]),
        "tau_min_ms": 1000.0 * float(open_loop_tau_min_s(drifted, line_speed_m_s=line_speed)),
        "speed_class": _speed_class(line_speed),
        "sigma_T_N": float(run.protocol.pct_T) * t_max,
        "sigma_v_m_s": float(run.protocol.pct_v) * v_max,
        "paper_reference": meta["paper_reference"],
        "R_m": list(drifted.roller_radius_m),
        "L_m": list(drifted.span_length_m),
        "J_kg_m2": list(drifted.inertia_kg_m2),
        "f_Nms_per_rad": list(drifted.kf),
    }
