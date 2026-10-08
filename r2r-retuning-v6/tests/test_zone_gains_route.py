"""POST /zone-gains (read) and POST /zone-gains/delete: the saved six gains of a plant.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_zone_gains_route.py -q
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.pipeline import zone_gains as Z
from backend.api.session_routes import _spec_or_error as from_dict_run

RUN_SPEC = {
    "plant": {"source": "preset", "preset_id": "P01"},
    "drift": {"EA_pct": -15.0, "J_UW_pct": 0.0, "J_Nip_pct": 0.0, "J_RW_pct": 0.0, "f_pct": 0.0},
    "protocol": {"T_log_ms": 5.0, "excitation": "E_Toggle", "record_s": 16.0, "pct_T": 0.003,
                 "pct_v": 0.003, "LPF_T_hz": 50.0, "LPF_v_hz": 50.0, "Kp_star": 100.0, "seed": 0},
}


@pytest.fixture(autouse=True)
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(Z, "STORE_DIR", tmp_path / "zone_gains")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_nothing_saved(client):
    r = client.post("/zone-gains", json={"run_spec": RUN_SPEC})
    assert r.status_code == 200 and r.json() == {"saved": None, "cost_tier": "T1"}


def test_reads_what_the_pipeline_saved(client):
    Z.save(from_dict_run(RUN_SPEC), "T1",
           {"kp_star_per_zone": [8.0, 4.1, 19.8], "ti_s_per_zone": [15.6, 18.6, 26.4]},
           source_run_hash="abc", S_twin=0.2)
    r = client.post("/zone-gains", json={"run_spec": RUN_SPEC})
    assert r.json()["saved"]["kp_star_per_zone"] == [8.0, 4.1, 19.8]


def test_delete(client):
    Z.save(from_dict_run(RUN_SPEC), "T1",
           {"kp_star_per_zone": [8.0, 4.1, 19.8], "ti_s_per_zone": [15.6, 18.6, 26.4]},
           source_run_hash="abc")
    r = client.post("/zone-gains/delete", json={"run_spec": RUN_SPEC})
    assert r.status_code == 200 and r.json() == {"deleted": True}
    assert client.post("/zone-gains", json={"run_spec": RUN_SPEC}).json()["saved"] is None


def test_bad_tier_is_422(client):
    r = client.post("/zone-gains", json={"run_spec": RUN_SPEC, "cost_tier": "T9"})
    assert r.status_code == 422
