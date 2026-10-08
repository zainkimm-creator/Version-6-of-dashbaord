"""Building a plant from an identified theta.

The estimator works in the paper's ratio parameters k_t = R^2/J and
k_f = f/J. Rebuilding a plant from them takes the roller geometry as known
from commissioning and inverts: J = R^2/k_t, then f = k_f * J. This mirrors
what identify_twin does internally.
"""

from __future__ import annotations

import pytest

from backend.pipeline.identify import identify
from backend.pipeline.payloads import DriftSpec, PlantSpec, ProtocolSpec, RunSpec
from backend.pipeline.plant import derive, params_from_spec
from backend.pipeline.twin import active_plant_spec, plant_spec_from_theta


def test_inverting_theta_true_returns_the_original_plant():
    base = PlantSpec(preset_id="P01")
    params, _ = params_from_spec(base)
    rebuilt_spec = plant_spec_from_theta(base, params.sysid_values())
    rebuilt, _ = params_from_spec(rebuilt_spec)
    assert rebuilt_spec.source == "custom"
    for actual, expected in zip(rebuilt.inertia_kg_m2, params.inertia_kg_m2):
        assert actual == pytest.approx(expected, rel=1e-12)
    for actual, expected in zip(rebuilt.kf, params.kf):
        assert actual == pytest.approx(expected, rel=1e-12)
    assert rebuilt.EA == pytest.approx(params.EA, rel=1e-12)
    assert rebuilt.roller_radius_m == params.roller_radius_m
    assert rebuilt.span_length_m == params.span_length_m
    assert rebuilt.feeder_velocity_m_s == params.feeder_velocity_m_s


def test_a_noise_free_identification_rebuilds_the_same_plant():
    base = PlantSpec(preset_id="P01")
    result = identify(
        RunSpec(base, DriftSpec(), ProtocolSpec(
            T_log_ms=1.0, pct_T=0.0, pct_v=0.0, LPF_T_hz=None, LPF_v_hz=None,
        ))
    )
    twin_spec = plant_spec_from_theta(base, result["estimates"])
    truth = derive(RunSpec(base, DriftSpec(), ProtocolSpec()))
    twin = derive(RunSpec(twin_spec, DriftSpec(), ProtocolSpec()))
    assert twin["zeta_cl_min"] == pytest.approx(truth["zeta_cl_min"], rel=1e-4)
    assert twin["tau_min_ms"] == pytest.approx(truth["tau_min_ms"], rel=1e-4)


def test_active_spec_returns_the_base_when_not_swapped():
    base = PlantSpec(preset_id="P01")
    assert active_plant_spec(base, {"EA": 1.0}, swapped=False) is base


def test_active_spec_returns_the_base_when_there_is_no_estimate():
    base = PlantSpec(preset_id="P01")
    assert active_plant_spec(base, None, swapped=True) is base


def test_active_spec_returns_the_twin_when_swapped():
    base = PlantSpec(preset_id="P01")
    params, _ = params_from_spec(base)
    swapped = active_plant_spec(base, params.sysid_values(), swapped=True)
    assert swapped.source == "custom"


def test_a_malformed_theta_hat_names_its_missing_keys():
    """`estimates` arrives from the client, so an incomplete one is a client
    mistake with a name, not a bare KeyError the route turns into a 500."""
    base = PlantSpec(preset_id="P01")
    params, _ = params_from_spec(base)
    partial = {k: v for k, v in params.sysid_values().items() if k not in ("EA", "kf_RW")}
    with pytest.raises(ValueError) as excinfo:
        plant_spec_from_theta(base, partial)
    assert "kf_RW" in str(excinfo.value)
    assert "EA" in str(excinfo.value)
