"""POST /retune: wiring, validation, and one real end-to-end run (slow)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

RUN_SPEC = {
    "plant": {"source": "preset", "preset_id": "P01"},
    "drift": {"EA_pct": 0.0, "J_UW_pct": -30.0, "J_Nip_pct": 0.0, "J_RW_pct": 50.0, "f_pct": 0.0},
    "protocol": {"T_log_ms": 5.0, "excitation": "E_Toggle", "record_s": 16.0, "pct_T": 0.003,
                 "pct_v": 0.003, "LPF_T_hz": 50.0, "LPF_v_hz": 50.0, "Kp_star": 100.0, "seed": 0},
}


# These end-to-end checks are about the paper's shared pair under Eq. (12). Since
# 7 Oct 2026 the defaults are T1 + the two-stage out-feeder, so they say so explicitly.
PAPER = {"gain_structure": "shared-2D", "cost_tier": "T0"}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_options_reach_the_pipeline(client, monkeypatch):
    from backend.pipeline import retune as retune_pipeline

    seen = {}

    def fake(run, options):
        seen["run"], seen["options"] = run, options
        return {"status": "ok"}

    monkeypatch.setattr(retune_pipeline, "retune", fake)
    body = client.post("/retune", json={"run_spec": RUN_SPEC, "bo_refine": True,
                                         "c_target_margin": 1.1}).json()
    assert body == {"status": "ok"}
    assert seen["options"].bo_refine is True
    assert seen["options"].c_target_margin == 1.1
    assert seen["run"].drift.J_RW_pct == 50.0


@pytest.mark.parametrize("field,value", [("epsilon_margin", 0), ("ti_flat_tolerance", 2.0)])
def test_nonsense_options_are_422(client, field, value):
    assert client.post("/retune", json={"run_spec": RUN_SPEC, field: value}).status_code == 422


def test_a_run_spec_missing_a_section_is_422(client):
    body = {"run_spec": {"plant": RUN_SPEC["plant"], "protocol": RUN_SPEC["protocol"]}}
    assert client.post("/retune", json=body).status_code == 422


def test_an_unbuildable_plant_is_400(client):
    """Same contract as /identify: a physically invalid plant names the field."""
    spec = {**RUN_SPEC, "plant": {"source": "custom"}}
    assert client.post("/retune", json={"run_spec": spec}).status_code == 400


@pytest.mark.slow
def test_a_real_retune_returns_gains_and_every_step(client):
    body = client.post("/retune", json={"run_spec": RUN_SPEC, **PAPER}).json()
    assert body["status"] == "ok"
    assert [s["step"] for s in body["steps"]] == [4, 5, 6, 7]
    delivered = body["gains"]["delivered"]
    assert 1.0 <= delivered["kp_star"] <= 500.0
    assert 0.5 <= delivered["ti_s"] <= 30.0
    assert delivered["ti_s"] > 0
    # The tie-break never lengthens T_I beyond the Eq. (12) optimum.
    assert body["gains"]["recommended"]["ti_s"] <= body["gains"]["hgs_only"]["ti_s"] + 1e-9
    # The true-plant reference is itself a grid search, so a transfer can land a hair below it.
    assert body["ratio_vs_reference"] >= 0.98
    assert body["eval_model"].startswith("author-step3")
    assert {a["id"] for a in body["assumptions"]} >= {"A1", "A3", "A6", "A7", "A8"}


@pytest.mark.slow
def test_a_real_retune_with_bo_refine_runs_end_to_end(client):
    """The BO branch bound its result to `run`, shadowing the RunSpec argument, and
    every bo_refine=True call died with AttributeError at the commissioning
    identification. Only a monkeypatched test covered this path, so it passed."""
    body = client.post("/retune", json={"run_spec": RUN_SPEC, "bo_refine": True, **PAPER}).json()
    assert body["status"] == "ok"
    bo = body["gains"]["hgs_bo"]
    assert bo is not None and bo["real_evaluations"] == 5
    assert 1.0 <= bo["kp_star"] <= 500.0 and 0.5 <= bo["ti_s"] <= 30.0
    # BO is only allowed to take over when it actually beats the twin recommendation.
    delivered = body["gains"]["delivered"]
    assert delivered["on_plant"]["S"] <= body["gains"]["recommended"]["on_plant"]["S"] + 1e-12


@pytest.mark.slow
def test_the_delivered_source_says_which_rule_produced_it(client):
    """With the tie-break off the delivered pair IS the twin optimum; saying it came
    from a 'T_I valley tie-break' described a rule that never ran."""
    from backend.pipeline import retune as retune_pipeline

    assert retune_pipeline.DEFAULT_TI_FLAT_TOLERANCE == 0.0
    body = client.post("/retune", json={"run_spec": RUN_SPEC, **PAPER}).json()
    # the twin's proposal carries the rule; step 7 may then restore today's gains (A9)
    assert body["gains"]["twin_candidate"]["source"] == "recommended (HGS-only, twin optimum as searched)"
    if not body["restore"]["restored"]:
        assert body["gains"]["delivered"]["source"] == body["gains"]["twin_candidate"]["source"]


def test_route_defaults_come_from_the_pipeline(client, monkeypatch):
    """A second copy of a default in the request model silently overrode the pipeline's."""
    from backend.pipeline import retune as retune_pipeline

    seen = {}
    monkeypatch.setattr(retune_pipeline, "retune",
                        lambda run, options: seen.setdefault("o", options) and None or {"status": "ok"})
    client.post("/retune", json={"run_spec": RUN_SPEC})
    assert seen["o"].ti_flat_tolerance == retune_pipeline.DEFAULT_TI_FLAT_TOLERANCE == 0.0
    assert seen["o"].epsilon_margin == 1.0 and seen["o"].c_target_margin == 1.0


@pytest.mark.slow
def test_a_live_retune_delivers_the_same_gains_as_the_offline_campaign(client):
    """The live loop and the §4.2 campaign must agree cell for cell on P001/D07."""
    import json
    from pathlib import Path

    cell_path = (Path(__file__).resolve().parents[1] / "reports" / "section4_author_spec"
                 / "cells" / "field_matched__P001__D07.json")
    if not cell_path.exists():
        pytest.skip("author-spec campaign not present")
    cell = json.loads(cell_path.read_text())
    spec = {**RUN_SPEC, "drift": {"EA_pct": 0.0, "J_UW_pct": -50.0, "J_Nip_pct": 0.0,
                                  "J_RW_pct": 100.0, "f_pct": 0.0}}
    body = client.post("/retune", json={"run_spec": spec, **PAPER}).json()
    delivered = body["gains"]["delivered"]
    assert delivered["kp_star"] == pytest.approx(cell["hgs"]["kp"], rel=1e-6)
    assert delivered["ti_s"] == pytest.approx(cell["hgs"]["ti_s"], rel=1e-6)
    assert delivered["on_plant"]["S"] == pytest.approx(cell["hgs"]["S_plant"], rel=1e-6)
