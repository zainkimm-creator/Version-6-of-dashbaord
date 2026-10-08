"""POST /plant/trace: the identification record the machine screen plays back."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.validation.plants import parameters_for_plant

PROTOCOL = {"T_log_ms": 5.0, "excitation": "E_Toggle", "record_s": 16.0, "pct_T": 0.003,
            "pct_v": 0.003, "LPF_T_hz": 50.0, "LPF_v_hz": 50.0, "Kp_star": 100.0, "seed": 0}
RUN = {"plant": {"source": "preset", "preset_id": "P01"},
       "drift": {"EA_pct": 0.0, "J_UW_pct": 0.0, "J_Nip_pct": 0.0, "J_RW_pct": 0.0, "f_pct": 0.0},
       "protocol": PROTOCOL}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_the_plant_plays_the_published_record(client):
    body = client.post("/plant/trace", json={"run_spec": RUN}).json()
    assert body["record_s"] == 16.0 and body["excitation"] == "E_Toggle"
    assert 0 < body["samples"] <= 600
    assert len(body["t_s"]) == len(body["T_N"]) == len(body["T_ref_N"]) == len(body["v_m_s"])
    assert all(math.isfinite(v) for row in body["T_N"] for v in row)
    base = body["T_ref_base_N"][0]
    # E_Toggle: span 1 steps +20 % at 1 s (supplement Table S1).
    after = [row[0] for t, row in zip(body["t_s"], body["T_ref_N"]) if 2.0 < t < 5.0]
    assert after and all(v == pytest.approx(1.2 * base) for v in after)
    assert body["twin_T_N"] is None


def test_an_exact_twin_plays_the_same_record(client):
    params, _ = parameters_for_plant("P01")
    body = client.post("/plant/trace", json={"run_spec": RUN,
                                             "estimates": params.sysid_values()}).json()
    assert body["twin_T_N"] is not None
    worst = max(abs(a - b) for pr, tr in zip(body["T_N"], body["twin_T_N"]) for a, b in zip(pr, tr))
    assert worst < 1e-3


def test_incomplete_estimates_are_422(client):
    assert client.post("/plant/trace", json={"run_spec": RUN, "estimates": {"EA": 1.0}}).status_code == 422


def test_a_multi_record_excitation_says_which_record(client):
    run = {**RUN, "protocol": {**PROTOCOL, "excitation": "ET3M", "record_s": 51.0}}
    body = client.post("/plant/trace", json={"run_spec": run}).json()
    assert body["records_total"] == 3 and body["note"]
