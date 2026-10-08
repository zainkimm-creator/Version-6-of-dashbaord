"""The session record and the staleness rules.

Staleness is derived from content hashes, never stored as a flag: an artefact
is stale exactly when the RunSpec that produced it differs from the current
one. Changing the plant, the drift or the protocol must invalidate
last_run -> theta_hat -> twin -> retune, with a reason naming what moved.
"""

from __future__ import annotations

import pytest

from backend.pipeline.payloads import DriftSpec, PlantSpec, ProtocolSpec, run_hash
from backend.session import store
from backend.session.state import SessionState, from_dict, staleness, to_dict


@pytest.fixture(autouse=True)
def temp_session(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SESSION_PATH", tmp_path / "current.json")
    yield


def identified_state() -> SessionState:
    from backend.pipeline.payloads import to_dict as spec_to_dict

    state = SessionState()
    key = state.current_hash
    origin = spec_to_dict(state.run_spec)
    state.last_run = {"run_hash": key, "n_samples": 3200, "run_spec": origin}
    state.theta_hat = {
        "run_hash": key, "MARE_theta_pct": 9.34, "converged": True, "run_spec": origin,
    }
    state.twin = {
        "loaded": True, "source_run_hash": key, "swapped": False, "run_spec": origin,
    }
    return state


def test_a_fresh_session_has_nothing_downstream():
    marks = staleness(SessionState())
    for name in ("last_run", "theta_hat", "twin", "retune"):
        assert marks[name]["present"] is False
        assert marks[name]["stale"] is False


def test_nothing_is_stale_immediately_after_identifying():
    marks = staleness(identified_state())
    for name in ("last_run", "theta_hat", "twin"):
        assert marks[name]["present"] is True
        assert marks[name]["stale"] is False


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda s: setattr(s, "plant", PlantSpec(preset_id="P07")), "plant changed"),
        (lambda s: setattr(s, "drift", DriftSpec(EA_pct=30.0)), "drift changed"),
        (lambda s: setattr(s, "protocol", ProtocolSpec(T_log_ms=20.0)), "protocol changed"),
    ],
)
def test_changing_any_input_invalidates_everything_downstream(mutate, reason):
    state = identified_state()
    mutate(state)
    marks = staleness(state)
    for name in ("last_run", "theta_hat", "twin"):
        assert marks[name]["stale"] is True, name
        assert marks[name]["reason"] == reason, name


def test_the_reason_names_every_field_that_moved():
    state = identified_state()
    state.plant = PlantSpec(preset_id="P07")
    state.drift = DriftSpec(EA_pct=10.0)
    assert staleness(state)["theta_hat"]["reason"] == "plant changed, drift changed"


def test_stale_artefacts_are_kept_not_deleted():
    state = identified_state()
    state.protocol = ProtocolSpec(T_log_ms=20.0)
    assert staleness(state)["theta_hat"]["stale"] is True
    assert state.theta_hat["MARE_theta_pct"] == 9.34


def test_current_hash_tracks_the_run_spec():
    state = SessionState()
    assert state.current_hash == run_hash(state.run_spec)


def test_round_trips_through_json():
    state = identified_state()
    assert from_dict(to_dict(state)) == state


def test_store_persists_and_reloads():
    state = identified_state()
    store.save(state)
    assert store.load() == state


def test_store_returns_a_default_session_when_none_exists():
    assert store.load() == SessionState()


def test_store_returns_a_default_session_when_the_file_is_corrupt():
    store.SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.SESSION_PATH.write_text("{not json", encoding="utf-8")
    assert store.load() == SessionState()


@pytest.mark.parametrize(
    "payload",
    [
        [1, 2, 3],
        "hello",
        42,
        None,
        {"plant": [1, 2, 3]},
        {"protocol": "not a mapping"},
    ],
    ids=[
        "top-level-list",
        "top-level-string",
        "top-level-number",
        "top-level-null",
        "plant-is-a-list",
        "protocol-is-a-string",
    ],
)
def test_store_returns_a_default_session_when_the_shape_is_wrong(payload):
    import json

    store.SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.SESSION_PATH.write_text(json.dumps(payload), encoding="utf-8")
    assert store.load() == SessionState()


def test_reset_clears_the_record():
    store.save(identified_state())
    assert store.reset() == SessionState()
    assert store.load() == SessionState()
