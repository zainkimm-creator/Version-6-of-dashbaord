"""The session API: the ten endpoints and the full walk through the loop."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.pipeline import cache
from backend.session import store


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SESSION_PATH", tmp_path / "current.json")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    yield


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


NOISE_FREE = {
    "T_log_ms": 1.0, "pct_T": 0.0, "pct_v": 0.0,
    "LPF_T_hz": None, "LPF_v_hz": None,
}


def run_spec(protocol_overrides=None, drift=None, plant=None):
    protocol = {
        "T_log_ms": 5.0, "excitation": "E_Toggle", "record_s": 16.0,
        "pct_T": 0.003, "pct_v": 0.003, "LPF_T_hz": 50.0, "LPF_v_hz": 50.0,
        "Kp_star": 100.0, "seed": 0,
    }
    protocol.update(protocol_overrides or {})
    return {
        "plant": plant or {"source": "preset", "preset_id": "P01"},
        "drift": drift or {},
        "protocol": protocol,
    }


def test_existing_routes_are_untouched(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert "plants" in client.get("/plants").json()


def test_fresh_session_is_the_default(client):
    body = client.get("/session").json()
    assert body["plant"]["preset_id"] == "P01"
    assert body["theta_hat"] is None
    assert body["staleness"]["theta_hat"]["present"] is False


def test_derive_returns_the_plant_readouts(client):
    body = client.post("/plant/derive", json={"run_spec": run_spec()}).json()
    assert body["regime"] in {"O-UD", "H-Osc", "H-Damp"}
    assert body["tau_min_ms"] > 0
    assert set(body["theta_true"]) == {
        "kt_UW", "kt_Nip", "kt_RW", "kf_UW", "kf_Nip", "kf_RW", "EA",
    }
    assert body["paper_reference"]["zeta_cl_min"] > 0


def test_derive_reports_not_published_for_a_custom_plant(client):
    custom = {
        "source": "custom", "preset_id": None,
        "R": [0.05, 0.05, 0.05], "L": [0.8, 0.8, 1.6],
        "J": [0.075, 0.055, 0.09], "f": [0.010, 0.012, 0.011],
        "EA": 4200.0, "v0": 1.0, "T_ref": 42.0,
    }
    body = client.post("/plant/derive", json={"run_spec": run_spec(plant=custom)}).json()
    assert body["paper_reference"] == {"status": "not_published"}


def test_invalid_custom_plant_is_a_400_naming_the_field(client):
    custom = {
        "source": "custom", "preset_id": None,
        "R": [0.05, 0.05, 0.05], "L": [0.8, 0.8, 1.6],
        "J": [0.075, 0.0, 0.09], "f": [0.010, 0.012, 0.011],
        "EA": 4200.0, "v0": 1.0, "T_ref": 42.0,
    }
    response = client.post("/plant/derive", json={"run_spec": run_spec(plant=custom)})
    assert response.status_code == 400
    assert "inertia" in response.json()["detail"]


def test_log_period_below_the_simulator_step_is_rejected(client):
    response = client.post(
        "/plant/derive", json={"run_spec": run_spec({"T_log_ms": 0.5})}
    )
    assert response.status_code == 422


# --- Fix round 1: dispatch on which spec constructor raised, not on the
# exception's message. A malformed protocol is always a 422, regardless of
# which of its fields is invalid. ---


@pytest.mark.parametrize(
    "overrides",
    [
        {"pct_T": -0.1},
        {"Kp_star": -5.0},
        {"LPF_T_hz": -1.0},
    ],
)
def test_invalid_protocol_fields_are_422_not_400(client, overrides):
    response = client.post(
        "/plant/derive", json={"run_spec": run_spec(overrides)}
    )
    assert response.status_code == 422


def test_identify_without_commit_leaves_the_session_alone(client):
    body = client.post(
        "/identify", json={"run_spec": run_spec(NOISE_FREE), "commit": False}
    ).json()
    assert body["converged"] is True
    assert body["MARE_theta_pct"] < 0.01
    assert client.get("/session").json()["theta_hat"] is None


def test_identify_with_commit_writes_the_session(client):
    client.patch("/session/protocol", json=NOISE_FREE)
    spec = client.get("/session").json()
    body = client.post(
        "/identify",
        json={
            "run_spec": {
                "plant": spec["plant"], "drift": spec["drift"], "protocol": spec["protocol"],
            },
            "commit": True,
        },
    ).json()
    session = client.get("/session").json()
    assert session["theta_hat"]["run_hash"] == body["run_hash"]
    assert session["staleness"]["theta_hat"]["stale"] is False


def test_presets_are_the_two_published_protocols(client):
    presets = client.get("/session/presets").json()["presets"]
    assert set(presets) == {"field_matched", "logging_only"}
    assert presets["field_matched"]["protocol"]["T_log_ms"] == pytest.approx(5.0)
    assert presets["logging_only"]["protocol"]["pct_v"] == pytest.approx(0.0)
    for preset in presets.values():
        assert preset["citation"]


def test_adopt_patches_only_the_named_field(client):
    before = client.get("/session").json()["protocol"]
    after = client.patch("/session/protocol", json={"T_log_ms": 20.0}).json()["protocol"]
    assert after["T_log_ms"] == 20.0
    assert after["excitation"] == before["excitation"]
    assert after["pct_T"] == before["pct_T"]


def test_the_whole_loop_and_its_invalidation(client):
    client.patch("/session/protocol", json=NOISE_FREE)
    spec = client.get("/session").json()
    identified = client.post(
        "/identify",
        json={
            "run_spec": {
                "plant": spec["plant"], "drift": spec["drift"], "protocol": spec["protocol"],
            },
            "commit": True,
        },
    ).json()

    loaded = client.put(
        "/session/twin", json={"action": "load", "run_hash": identified["run_hash"]}
    ).json()
    assert loaded["twin"]["loaded"] is True
    assert loaded["staleness"]["twin"]["stale"] is False

    after_drift = client.put("/session/drift", json={"EA_pct": 30.0}).json()
    for name in ("last_run", "theta_hat", "twin"):
        assert after_drift["staleness"][name]["stale"] is True, name
        assert after_drift["staleness"][name]["reason"] == "drift changed", name

    assert after_drift["theta_hat"]["MARE_theta_pct"] == identified["MARE_theta_pct"]


def test_swap_flips_the_flag_and_survives_a_reload(client):
    client.patch("/session/protocol", json=NOISE_FREE)
    spec = client.get("/session").json()
    identified = client.post(
        "/identify",
        json={
            "run_spec": {
                "plant": spec["plant"], "drift": spec["drift"], "protocol": spec["protocol"],
            },
            "commit": True,
        },
    ).json()
    client.put("/session/twin", json={"action": "load", "run_hash": identified["run_hash"]})
    swapped = client.put("/session/twin", json={"action": "swap"}).json()
    assert swapped["twin"]["swapped"] is True
    assert client.get("/session").json()["twin"]["swapped"] is True


def test_swap_without_a_twin_is_a_409(client):
    response = client.put("/session/twin", json={"action": "swap"})
    assert response.status_code == 409


def test_load_without_a_committed_identification_is_a_409(client):
    response = client.put("/session/twin", json={"action": "load"})
    assert response.status_code == 409
    assert "identify and commit" in response.json()["detail"]


def test_load_with_a_run_hash_that_is_not_the_committed_one_is_a_409(client):
    """The provenance field exists to say which identification the twin came
    from; it must not accept an arbitrary caller-supplied string."""
    client.patch("/session/protocol", json=NOISE_FREE)
    spec = client.get("/session").json()
    client.post(
        "/identify",
        json={
            "run_spec": {
                "plant": spec["plant"], "drift": spec["drift"], "protocol": spec["protocol"],
            },
            "commit": True,
        },
    )
    response = client.put(
        "/session/twin", json={"action": "load", "run_hash": "totally-bogus-hash"}
    )
    assert response.status_code == 409
    assert client.get("/session").json()["twin"] is None


def test_reset_clears_everything(client):
    client.put("/session/drift", json={"EA_pct": 30.0})
    body = client.post("/session/reset").json()
    assert body["drift"]["EA_pct"] == 0.0
    assert body["theta_hat"] is None


def test_derive_is_pure_and_takes_swap_as_an_input(client):
    """Swapping must not depend on session state the body does not carry."""
    truth = client.post("/plant/derive", json={"run_spec": run_spec()}).json()
    assert truth["derived_from"] == "theta_true"

    estimates = {
        name: value * 1.10 for name, value in truth["theta_true"].items()
    }
    swapped = client.post(
        "/plant/derive",
        json={"run_spec": run_spec(), "swap_with_estimates": estimates},
    ).json()
    assert swapped["derived_from"] == "theta_hat"
    assert swapped["theta_true"]["EA"] == pytest.approx(estimates["EA"])
    assert swapped["zeta_cl_min"] != truth["zeta_cl_min"]


def test_paper_check_reports_not_published_for_an_unpublished_protocol(client):
    body = client.post(
        "/identify/paper-check", json={"run_spec": run_spec({"T_log_ms": 37.0})}
    ).json()
    assert body["status"] == "not_published"


def test_paper_check_finds_the_field_matched_cell(client):
    body = client.post("/identify/paper-check", json={"run_spec": run_spec()}).json()
    assert body["status"] == "published"
    assert body["preset"] == "field_matched"


# --- Added beyond the brief's 17: non-convergence must never surface as a
# large number or an error. `identify` reports it as converged=False with a
# null MARE and HTTP 200; nothing else in the test suite exercises that path
# at the route level. ---


def test_identify_non_convergence_is_a_200_not_a_large_number(client, monkeypatch):
    from backend.api import session_routes

    def fake_identify(run, *, use_cache: bool = True):
        return {
            "run_hash": "deadbeefcafef00d",
            "plant_id": "P01",
            "rows": [],
            "estimates": None,
            "MARE_theta_pct": None,
            "converged": False,
            "optimizer_status": "did not converge",
            "nfev": 500,
            "max_nfev": 500,
            "effective_record_s": 16.0,
            "n_samples": 3200,
            "excitation": "E_Toggle",
            "Kp_star": 100.0,
            "produced_at": "2026-09-07T00:00:00+00:00",
            "cached": False,
        }

    monkeypatch.setattr(session_routes.identify_pipeline, "identify", fake_identify)

    response = client.post(
        "/identify", json={"run_spec": run_spec(NOISE_FREE), "commit": True}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["converged"] is False
    assert body["MARE_theta_pct"] is None

    # commit=True must also write the non-convergent result back honestly,
    # never silently dropping the failure or fabricating a number.
    session = client.get("/session").json()
    assert session["theta_hat"]["converged"] is False
    assert session["theta_hat"]["MARE_theta_pct"] is None


# --- Fix wave: swap must not apply the drift twice. theta-hat is fitted
# against the *drifted* plant, so the twin rebuilt from it already embodies
# the drift; the effective run's drift has to be identity or `plant.derive`
# drifts an already-drifted plant. Every earlier swap test ran at zero drift,
# where identity and `run.drift` are the same thing, so none of them could
# see this. ---


def test_swap_under_drift_does_not_apply_the_drift_twice(client):
    """At non-zero drift the swapped readouts must still describe the same plant.

    Noise-free 1 ms logging makes the fit essentially exact (MARE_theta of
    order 1e-6 %), so theta-hat *is* the drifted plant to within float noise
    and the swapped and unswapped answers must agree. The tolerance below is
    ~250x the residual actually observed (rel 3.9e-9 on zeta) and still four
    orders of magnitude tighter than the double-drift error it guards against
    (6.6 % on zeta, 10.6 % on tau_min).
    """
    drift = {"EA_pct": 30.0}
    spec = run_spec(NOISE_FREE, drift=drift)

    identified = client.post("/identify", json={"run_spec": spec, "commit": True}).json()
    assert identified["converged"] is True
    assert identified["MARE_theta_pct"] < 0.01

    unswapped = client.post("/plant/derive", json={"run_spec": spec}).json()
    swapped = client.post(
        "/plant/derive",
        json={"run_spec": spec, "swap_with_estimates": identified["estimates"]},
    ).json()

    assert unswapped["drift_applied"] is True
    # The twin *is* the drifted plant, so there is no second drift to show.
    assert swapped["drift_applied"] is False
    assert swapped["theta_true"]["EA"] == pytest.approx(
        unswapped["theta_drifted"]["EA"], rel=1e-6
    )

    assert swapped["zeta_cl_min"] == pytest.approx(unswapped["zeta_cl_min"], rel=1e-6)
    assert swapped["tau_min_ms"] == pytest.approx(unswapped["tau_min_ms"], rel=1e-6)


# --- Fix wave: the spec constructors coerce with float()/int(), so
# well-formed JSON carrying a wrong-typed field raises TypeError, which none
# of the handlers caught -- it escaped as a 500. A wrong type is a client
# mistake and keeps the same 400-vs-422 split as a bad value. ---


def test_wrong_typed_drift_field_is_a_400_not_a_500(client):
    response = client.put("/session/drift", json={"EA_pct": None})
    assert response.status_code == 400


def test_wrong_typed_protocol_field_is_a_422_not_a_500(client):
    response = client.patch("/session/protocol", json={"T_log_ms": None})
    assert response.status_code == 422


def test_wrong_typed_plant_field_is_a_400_not_a_500(client):
    """`R` must be a triple; a bare number is not iterable, so `_triple` raises
    TypeError rather than the ValueError a wrong-length triple gives."""
    custom = {
        "source": "custom", "preset_id": None,
        "R": 5, "L": [0.8, 0.8, 1.6],
        "J": [0.075, 0.055, 0.09], "f": [0.010, 0.012, 0.011],
        "EA": 4200.0, "v0": 1.0, "T_ref": 42.0,
    }
    response = client.post("/plant/derive", json={"run_spec": run_spec(plant=custom)})
    assert response.status_code == 400


def test_derive_with_an_incomplete_swap_payload_is_a_400_not_a_500(client):
    truth = client.post("/plant/derive", json={"run_spec": run_spec()}).json()
    partial = {k: v for k, v in truth["theta_true"].items() if k != "EA"}
    response = client.post(
        "/plant/derive",
        json={"run_spec": run_spec(), "swap_with_estimates": partial},
    )
    assert response.status_code == 400
    assert "EA" in response.json()["detail"]
