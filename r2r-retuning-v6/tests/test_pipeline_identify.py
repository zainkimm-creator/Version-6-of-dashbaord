"""The identify pipeline: RunSpec in, two-column panel out.

The round-trip test is the load-bearing one. At T_log = T_s the one-step
prediction is exact, so a noise-free identification must return theta_true.
It is not a vacuous check: the same setup at T_log = 5 ms gives about 2.6 %.
"""

from __future__ import annotations

import pytest

from backend.pipeline import cache
from backend.pipeline.identify import identify, protocol_to_paper_protocol
from backend.pipeline.payloads import DriftSpec, PlantSpec, ProtocolSpec, RunSpec, run_hash

PARAMETERS = ("kt_UW", "kt_Nip", "kt_RW", "kf_UW", "kf_Nip", "kf_RW", "EA")


@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    yield


def noise_free_run(tlog_ms: float) -> RunSpec:
    return RunSpec(
        PlantSpec(preset_id="P01"),
        DriftSpec(),
        ProtocolSpec(
            T_log_ms=tlog_ms,
            pct_T=0.0,
            pct_v=0.0,
            LPF_T_hz=None,
            LPF_v_hz=None,
            record_s=16.0,
            seed=0,
        ),
    )


def test_round_trip_recovers_theta_true():
    """Required test #2 from the spec."""
    result = identify(noise_free_run(1.0))
    assert result["converged"] is True
    assert result["MARE_theta_pct"] < 0.01


def test_round_trip_threshold_is_not_vacuous():
    coarse = identify(noise_free_run(5.0))
    assert coarse["MARE_theta_pct"] > 0.5


def test_protocol_spec_maps_onto_the_paper_protocol():
    spec = ProtocolSpec(
        T_log_ms=20.0, pct_T=0.003, pct_v=0.0, LPF_T_hz=100.0,
        LPF_v_hz=None, record_s=16.0,
    )
    protocol = protocol_to_paper_protocol(spec)
    assert protocol.log_sample_time_s == pytest.approx(0.020)
    assert protocol.tension_noise_fraction == pytest.approx(0.003)
    assert protocol.velocity_noise_fraction == pytest.approx(0.0)
    assert protocol.tension_lpf_hz == 100.0
    assert protocol.velocity_lpf_hz is None
    assert protocol.record_duration_s == pytest.approx(16.0)


def test_rows_carry_all_seven_parameters_in_order():
    result = identify(noise_free_run(1.0))
    assert [row["parameter"] for row in result["rows"]] == list(PARAMETERS)
    for row in result["rows"]:
        assert set(row) == {"parameter", "theta_true", "theta_hat", "error_pct"}


def test_result_carries_the_run_hash():
    run = noise_free_run(1.0)
    assert identify(run)["run_hash"] == run_hash(run)


def test_second_call_is_served_from_cache():
    run = noise_free_run(1.0)
    first = identify(run)
    second = identify(run)
    assert first["cached"] is False
    assert second["cached"] is True
    assert {k: v for k, v in first.items() if k != "cached"} == {
        k: v for k, v in second.items() if k != "cached"
    }


def test_use_cache_false_recomputes():
    run = noise_free_run(1.0)
    identify(run)
    assert identify(run, use_cache=False)["cached"] is False


def test_drift_changes_the_result_and_the_hash():
    base = noise_free_run(1.0)
    drifted = RunSpec(base.plant, DriftSpec(EA_pct=30.0), base.protocol)
    assert identify(drifted)["run_hash"] != identify(base)["run_hash"]
    assert identify(drifted)["MARE_theta_pct"] < 0.01  # noise-free is still exact


def test_error_percentages_agree_with_the_two_columns():
    result = identify(
        RunSpec(PlantSpec(preset_id="P01"), DriftSpec(EA_pct=30.0), ProtocolSpec())
    )
    for row in result["rows"]:
        expected = 100.0 * (row["theta_hat"] - row["theta_true"]) / row["theta_true"]
        assert row["error_pct"] == pytest.approx(expected, rel=1e-9)


def test_custom_plant_identifies():
    spec = PlantSpec(
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
    result = identify(
        RunSpec(spec, DriftSpec(), ProtocolSpec(
            T_log_ms=1.0, pct_T=0.0, pct_v=0.0, LPF_T_hz=None, LPF_v_hz=None,
        ))
    )
    assert result["converged"] is True
    assert result["MARE_theta_pct"] < 0.01


def test_non_convergence_nulls_the_result_instead_of_reporting_a_number(monkeypatch):
    """Rule 4 at the pipeline, not just at the route.

    The route-level test hands `identify` a hand-written payload, so it proves
    nothing about `identify`'s own converged-ternaries. This is their witness:
    a diverged fit is no estimate, so there are no rows, no estimates and no
    MARE -- never a large number standing in for a failure.
    """
    from backend.pipeline import identify as identify_module

    def fake_identify_twin(params, protocol, plant_meta, **kwargs):
        return None, {
            "converged": False,
            "estimates": {name: 1e9 for name in PARAMETERS},
            "mare_theta_percent": 4.2e6,
            "optimizer_status": "max nfev reached",
            "nfev": 500,
            "max_nfev": 500,
            "excitation": kwargs.get("excitation"),
            "kp_star": kwargs.get("kp_star"),
        }

    monkeypatch.setattr(identify_module, "identify_twin", fake_identify_twin)

    result = identify(noise_free_run(1.0))
    assert result["converged"] is False
    assert result["MARE_theta_pct"] is None
    assert result["rows"] == []
    assert result["estimates"] is None
