"""GET /atlas — the route the Twin Study screen reads."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.atlas import reader
from backend.pipeline.payloads import paper_record_s


@pytest.fixture(autouse=True)
def atlas(tmp_path, monkeypatch):
    cells = tmp_path / "cells"
    (cells / "twin").mkdir(parents=True)
    (cells / "reference").mkdir(parents=True)
    (cells / "twin" / "abc123.json").write_text(json.dumps({
        "atlas_version": reader.ATLAS_VERSION,
        "kind": "twin", "run_hash": "abc123", "plant_id": "P01",
        "T_log_ms": 5.0, "excitation": "E_Toggle",
        "LPF_T_hz": 50.0, "LPF_v_hz": 50.0, "converged": True,
        "MARE_theta_pct": 3.61,
        "gains": {"kp_star": 2.07, "ti_s": 1853.4, "kp_on_bound": None,
                  "ti_scale_on_bound": "upper", "plant_auto_ti_s": 18.534},
        "S_achieved_on_physical": 2.0,
    }), encoding="utf-8")
    (cells / "reference" / "P01.json").write_text(json.dumps({
        "atlas_version": reader.ATLAS_VERSION,
        "kind": "reference", "plant_id": "P01", "auto_ti_s": 18.534,
        "full": {"kp_star": 2.0, "ti_s": 1853.4, "S": 1.6, "evaluations": 2805},
        "coarse": {"kp_star": 2.07, "ti_s": 1853.4, "S": 2.0, "evaluations": 100},
    }), encoding="utf-8")
    monkeypatch.setattr(reader, "CELL_DIR", cells)
    monkeypatch.setattr(reader, "ATLAS_DIR", tmp_path)
    reader.twin_index.cache_clear()
    yield
    reader.twin_index.cache_clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def pending_reference_atlas(tmp_path, monkeypatch):
    """A twin cell whose plant's reference has not been computed yet.

    A separate, self-contained atlas (not a mutation of the shared `atlas`
    fixture's files) so no state can leak between tests: it builds its own
    cell tree and points `reader.CELL_DIR`/`reader.ATLAS_DIR` at it.
    """
    cells = tmp_path / "pending"
    (cells / "twin").mkdir(parents=True)
    (cells / "reference").mkdir(parents=True)
    (cells / "twin" / "abc123.json").write_text(json.dumps({
        "atlas_version": reader.ATLAS_VERSION,
        "kind": "twin", "run_hash": "abc123", "plant_id": "P01",
        "T_log_ms": 5.0, "excitation": "E_Toggle",
        "LPF_T_hz": 50.0, "LPF_v_hz": 50.0, "converged": True,
        "MARE_theta_pct": 3.61,
        "gains": {"kp_star": 2.07, "ti_s": 1853.4, "kp_on_bound": None,
                  "ti_scale_on_bound": "upper", "plant_auto_ti_s": 18.534},
        "S_achieved_on_physical": 2.0,
    }), encoding="utf-8")
    # No reference/P01.json written: the generator writes twin cells before
    # their plant's reference is ready, and this is the status that produces.
    monkeypatch.setattr(reader, "CELL_DIR", cells)
    monkeypatch.setattr(reader, "ATLAS_DIR", tmp_path)
    reader.twin_index.cache_clear()
    yield
    reader.twin_index.cache_clear()


@pytest.fixture
def not_converged_atlas(tmp_path, monkeypatch):
    """A twin cell whose identification failed under this protocol."""
    cells = tmp_path / "dead"
    (cells / "twin").mkdir(parents=True)
    (cells / "reference").mkdir(parents=True)
    (cells / "twin" / "dead456.json").write_text(json.dumps({
        "atlas_version": reader.ATLAS_VERSION,
        "kind": "twin", "run_hash": "dead456", "plant_id": "P01",
        "T_log_ms": 1.0, "excitation": "ET1",
        "LPF_T_hz": 50.0, "LPF_v_hz": 50.0, "converged": False,
        "MARE_theta_pct": None,
    }), encoding="utf-8")
    monkeypatch.setattr(reader, "CELL_DIR", cells)
    monkeypatch.setattr(reader, "ATLAS_DIR", tmp_path)
    reader.twin_index.cache_clear()
    yield
    reader.twin_index.cache_clear()


ON_GRID = {"plant_id": "P01", "T_log_ms": 5.0, "excitation": "E_Toggle",
           "record_s": 16.0, "pct_T": 0.003, "pct_v": 0.003,
           "LPF_T_hz": 50.0, "LPF_v_hz": 50.0, "Kp_star": 100.0, "seed": 0}


def on_grid(**over):
    """ON_GRID with an override, keeping `record_s` tied to the excitation.

    The two are not independent: the atlas holds each excitation at its own
    published duration, so swapping the excitation alone would put the request
    off-grid on `record_s` instead of testing what the test names.
    """
    params = {**ON_GRID, **over}
    if "excitation" in over and "record_s" not in over:
        params["record_s"] = paper_record_s(over["excitation"], params["pct_v"])
    return params


def test_a_precomputed_cell_returns_both_ratios(client):
    body = client.get("/atlas", params=ON_GRID).json()
    assert body["status"] == "ok"
    assert body["cell"]["run_hash"] == "abc123"
    assert body["ratio_vs_full"] == pytest.approx(2.0 / 1.6)
    assert body["ratio_vs_coarse"] == pytest.approx(1.0)


def test_an_off_grid_protocol_is_not_precomputed(client):
    body = client.get("/atlas", params={**ON_GRID, "LPF_T_hz": 75.0,
                                        "LPF_v_hz": 75.0}).json()
    assert body["status"] == "not_precomputed"
    assert body["off_grid_field"] == "LPF_T_hz"


def test_a_missing_cell_never_returns_a_neighbour(client):
    body = client.get("/atlas", params={**ON_GRID, "T_log_ms": 20.0}).json()
    assert body["status"] == "not_precomputed"
    assert "cell" not in body


def test_boundary_flags_survive_the_route(client):
    body = client.get("/atlas", params=ON_GRID).json()
    assert body["cell"]["gains"]["ti_scale_on_bound"] == "upper"


def test_the_route_is_a_200_in_every_status(client):
    for params in (ON_GRID,
                   {**ON_GRID, "LPF_T_hz": 75.0, "LPF_v_hz": 75.0},
                   {**ON_GRID, "T_log_ms": 20.0}):
        assert client.get("/atlas", params=params).status_code == 200


def test_existing_routes_are_untouched(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert "plants" in client.get("/plants").json()
    assert "GET /atlas" in client.get("/metadata").json()["routes"]


def test_a_pending_reference_returns_null_ratios_never_a_number(client, pending_reference_atlas):
    response = client.get("/atlas", params=ON_GRID)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "reference_pending"
    assert body["cell"]["run_hash"] == "abc123"
    assert body["ratio_vs_full"] is None
    assert body["ratio_vs_coarse"] is None


def test_a_non_converged_cell_returns_null_ratios_at_the_route(client, not_converged_atlas):
    response = client.get(
        "/atlas", params=on_grid(T_log_ms=1.0, excitation="ET1")
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_converged"
    assert body["ratio_vs_full"] is None
    assert body["ratio_vs_coarse"] is None


def test_an_empty_plant_id_is_not_precomputed_not_a_500(client):
    response = client.get("/atlas", params={**ON_GRID, "plant_id": ""})
    assert response.status_code == 200
    assert response.json()["status"] == "not_precomputed"


def test_an_unfiltered_protocol_is_on_the_grid_not_rejected(client):
    """Clearing the LPF box is a real protocol -- no filter -- and the atlas
    now holds it.

    It must survive the query string as None and reach the grid check without a
    422; "none" is one of the paper's own conditions (Fig. S6 panel a). It is
    reported off-grid only on the CELL's absence from this tiny fixture, never
    on the field, which is the difference between "we do not hold that cutoff"
    and "we have not computed that cell".
    """
    for raw in ("none", "", "null", "NONE"):
        response = client.get(
            "/atlas", params={**ON_GRID, "LPF_T_hz": raw, "LPF_v_hz": raw}
        )
        assert response.status_code == 200, raw
        body = response.json()
        assert body["off_grid_field"] is None, raw


def test_a_null_velocity_lpf_names_its_own_field(client):
    body = client.get("/atlas", params={**ON_GRID, "LPF_v_hz": "none"}).json()
    assert body["status"] == "not_precomputed"
    assert body["off_grid_field"] == "LPF_v_hz"


def test_a_genuinely_malformed_lpf_is_still_a_422(client):
    assert client.get("/atlas", params={**ON_GRID, "LPF_T_hz": "abc"}).status_code == 422


def test_an_omitted_lpf_still_resolves_the_50_hz_cell(client):
    """The default must stay 50 Hz -- but only when the caller said nothing.

    Answering a *null* LPF with this cell would be inventing a value, which is
    why the null case above is off-grid rather than defaulted.
    """
    params = {k: v for k, v in ON_GRID.items() if not k.startswith("LPF_")}
    body = client.get("/atlas", params=params).json()
    assert body["status"] == "ok"
    assert body["cell"]["run_hash"] == "abc123"


def test_a_stale_atlas_version_cell_is_not_served(client, tmp_path, monkeypatch):
    cells = tmp_path / "stale"
    (cells / "twin").mkdir(parents=True)
    (cells / "reference").mkdir(parents=True)
    (cells / "twin" / "old.json").write_text(json.dumps({
        "atlas_version": "gain_atlas_v0",
        "kind": "twin", "run_hash": "old", "plant_id": "P01",
        "T_log_ms": 5.0, "excitation": "E_Toggle", "converged": True,
        "MARE_theta_pct": 3.61, "S_achieved_on_physical": 2.0,
        "gains": {"kp_star": 2.07, "ti_s": 1853.4},
    }), encoding="utf-8")
    monkeypatch.setattr(reader, "CELL_DIR", cells)
    monkeypatch.setattr(reader, "ATLAS_DIR", tmp_path)
    reader.twin_index.cache_clear()
    try:
        response = client.get("/atlas", params=ON_GRID)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "not_precomputed"
        assert "cell" not in body
    finally:
        reader.twin_index.cache_clear()


def test_the_response_carries_how_much_of_the_atlas_exists(client):
    """`not_precomputed` with a null field is "not yet", and the screen says so."""
    body = client.get("/atlas", params={**ON_GRID, "T_log_ms": 20.0}).json()
    assert body["status"] == "not_precomputed"
    assert body["off_grid_field"] is None
    assert body["cell_counts"]["twin"] == 1
    assert body["cell_counts"]["reference"] == 1
