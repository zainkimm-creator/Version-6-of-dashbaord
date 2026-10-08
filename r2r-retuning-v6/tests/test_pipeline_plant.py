"""Plant construction from a PlantSpec, and the derived readouts.

The preset path must be indistinguishable from the existing
`parameters_for_plant`, because every frozen study number depends on it.
"""

from __future__ import annotations

import pytest

from backend.models.modal import open_loop_tau_min_s
from backend.pipeline.payloads import DriftSpec, PlantSpec, ProtocolSpec, RunSpec
from backend.pipeline.plant import derive, drifted_params, params_from_spec
from backend.validation.plants import parameters_for_plant, plant_registry


def test_preset_path_is_identical_to_parameters_for_plant():
    for row in plant_registry():
        plant_id = str(row["plant_id"])
        expected, _ = parameters_for_plant(plant_id)
        actual, _ = params_from_spec(PlantSpec(preset_id=plant_id))
        assert actual == expected, plant_id


def test_custom_path_builds_the_requested_plant():
    spec = PlantSpec(
        source="custom",
        preset_id=None,
        R=(0.05, 0.06, 0.07),
        L=(0.8, 0.9, 1.6),
        J=(0.075, 0.055, 0.09),
        f=(0.010, 0.012, 0.011),
        EA=4200.0,
        v0=1.25,
        T_ref=42.0,
    )
    params, meta = params_from_spec(spec)
    assert params.roller_radius_m == (0.05, 0.06, 0.07)
    assert params.span_length_m == (0.8, 0.9, 1.6)
    assert params.EA == 4200.0
    assert params.feeder_velocity_m_s == 1.25
    assert params.tension_ref_N == (42.0, 42.0, 42.0)
    assert meta["paper_reference"] == {"status": "not_published"}
    assert meta["T_max_N"] == pytest.approx(42.0 / 0.30)
    assert meta["v_max_m_s"] == pytest.approx(1.25 / 0.30)


@pytest.mark.parametrize(
    "field, value",
    [("EA", 0.0), ("v0", -1.0), ("T_ref", 0.0)],
)
def test_custom_plant_rejects_non_positive_scalars(field, value):
    kwargs = dict(
        source="custom",
        preset_id=None,
        R=(0.05, 0.05, 0.05),
        L=(0.8, 0.8, 1.6),
        J=(0.075, 0.055, 0.09),
        f=(0.010, 0.012, 0.011),
        EA=4200.0,
        v0=1.0,
        T_ref=42.0,
    )
    kwargs[field] = value
    with pytest.raises(ValueError):
        params_from_spec(PlantSpec(**kwargs))


def test_custom_plant_rejects_non_positive_inertia():
    with pytest.raises(ValueError, match="inertia"):
        params_from_spec(
            PlantSpec(
                source="custom",
                preset_id=None,
                R=(0.05, 0.05, 0.05),
                L=(0.8, 0.8, 1.6),
                J=(0.075, 0.0, 0.09),
                f=(0.010, 0.012, 0.011),
                EA=4200.0,
                v0=1.0,
                T_ref=42.0,
            )
        )


def test_drift_percentages_become_multipliers():
    base, drifted, _ = drifted_params(
        PlantSpec(preset_id="P01"), DriftSpec(EA_pct=30.0, J_UW_pct=-30.0, f_pct=15.0)
    )
    assert drifted.EA == pytest.approx(base.EA * 1.30)
    assert drifted.inertia_kg_m2[0] == pytest.approx(base.inertia_kg_m2[0] * 0.70)
    assert drifted.kf_UW == pytest.approx(base.kf_UW * 1.15)
    assert drifted.inertia_kg_m2[2] == pytest.approx(base.inertia_kg_m2[2])


def test_zero_drift_leaves_the_plant_untouched():
    base, drifted, _ = drifted_params(PlantSpec(preset_id="P05"), DriftSpec())
    assert base == drifted


def test_derive_matches_the_published_damping_and_tau_for_every_preset():
    for row in plant_registry():
        plant_id = str(row["plant_id"])
        params, _ = parameters_for_plant(plant_id)
        payload = derive(
            RunSpec(PlantSpec(preset_id=plant_id), DriftSpec(), ProtocolSpec())
        )
        assert payload["zeta_cl_min"] == pytest.approx(
            float(row["zeta_cl_min"]), abs=0.02
        ), plant_id
        assert payload["tau_min_ms"] == pytest.approx(
            1000.0 * open_loop_tau_min_s(params, line_speed_m_s=float(row["v_ref_m_s"])),
            rel=1e-9,
        ), plant_id
        assert payload["regime"] in {"O-UD", "H-Osc", "H-Damp"}
        assert payload["paper_reference"]["zeta_cl_min"] == pytest.approx(
            float(row["zeta_cl_min"])
        )


def test_derive_reports_the_noise_doses_from_the_protocol():
    payload = derive(
        RunSpec(
            PlantSpec(preset_id="P01"),
            DriftSpec(),
            ProtocolSpec(pct_T=0.003, pct_v=0.006),
        )
    )
    t_ref = payload["T_ref_N"]
    v0 = payload["v0_m_s"]
    assert payload["sigma_T_N"] == pytest.approx(0.003 * t_ref / 0.30)
    assert payload["sigma_v_m_s"] == pytest.approx(0.006 * v0 / 0.30)


def test_derive_classifies_line_speed():
    def speed_class(v0):
        spec = PlantSpec(
            source="custom",
            preset_id=None,
            R=(0.05, 0.05, 0.05),
            L=(0.8, 0.8, 1.6),
            J=(0.075, 0.055, 0.09),
            f=(0.010, 0.012, 0.011),
            EA=4200.0,
            v0=v0,
            T_ref=42.0,
        )
        return derive(RunSpec(spec, DriftSpec(), ProtocolSpec()))["speed_class"]

    assert speed_class(0.4) == "slow"
    assert speed_class(0.5) == "slow"
    assert speed_class(1.0) == "mid"
    assert speed_class(2.0) == "fast"
    assert speed_class(3.0) == "fast"


def test_derive_returns_all_seven_parameters_for_true_and_drifted():
    payload = derive(
        RunSpec(PlantSpec(preset_id="P01"), DriftSpec(EA_pct=30.0), ProtocolSpec())
    )
    names = ("kt_UW", "kt_Nip", "kt_RW", "kf_UW", "kf_Nip", "kf_RW", "EA")
    assert tuple(payload["theta_true"]) == names
    assert tuple(payload["theta_drifted"]) == names
    assert payload["theta_drifted"]["EA"] == pytest.approx(
        payload["theta_true"]["EA"] * 1.30
    )
