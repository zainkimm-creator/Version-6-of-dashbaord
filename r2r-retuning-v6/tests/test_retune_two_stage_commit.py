"""When the six gains are written to the store (fast; no simulation).

The six gains are saved only for a case whose result the line accepted (step 7 did
not restore), and a cache hit re-commits them -- so FORGET followed by the same click
saves them again instead of claiming "saved" over an empty store.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_retune_two_stage_commit.py -q
"""

from __future__ import annotations

import copy

import pytest

from backend.pipeline import retune as R
from backend.pipeline import zone_gains as Z
from backend.pipeline.payloads import DEFAULT_RUN

SIX = {"kp_star_per_zone": [6.5, 3.9, 20.3], "ti_s_per_zone": [12.0, 30.0, 11.4]}


@pytest.fixture(autouse=True)
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(Z, "STORE_DIR", tmp_path / "zone_gains")


def _payload(s_candidate, s_commissioned, *, reused=False, saved=True):
    def block(kp, ti, s):
        return {"kp_star": None, "ti_s": None, "kp_star_per_zone": kp, "ti_s_per_zone": ti,
                "per_zone": True, "on_plant": {"S": s}}

    return {
        "status": "ok", "run_hash": "case-1", "cost_tier": "T1",
        "steps": [{"step": 7, "name": "Validate and restore", "status": "fail", "detail": ""}],
        "search": {"per_zone_gains": {**copy.deepcopy(SIX), "S_twin": 0.6, "twin_evals": 10, "reused": reused,
                                      "saved": saved, "source_run_hash": "case-1"}},
        "gains": {"delivered": block([6.5, 4.2, 20.3], [12.0, 30.0, 11.4], s_candidate),
                  "commissioned": block([5.7] * 3, [30.0] * 3, s_commissioned),
                  "reference_on_plant": block([1, 1, 1], [1, 1, 1], 0.5)},
        "acceptance": {"S_delivered": s_candidate, "C_target": 0.9, "margin": 1.0, "passed": True},
        "ratio_vs_reference": None,
    }


def test_accepted_case_saves_the_six():
    out = R._commit_per_zone(DEFAULT_RUN, R.RetuneOptions(), _payload(0.6, 0.98, saved=None))
    assert out["search"]["per_zone_gains"]["saved"] is True
    assert Z.load(DEFAULT_RUN, "T1")["kp_star_per_zone"] == SIX["kp_star_per_zone"]


def test_restored_case_does_not_save_the_six():
    out = R._commit_per_zone(DEFAULT_RUN, R.RetuneOptions(), _payload(1.2, 0.98, saved=None))
    assert out["search"]["per_zone_gains"]["saved"] is False
    assert "restored" in out["search"]["per_zone_gains"]["not_saved_because"]
    assert Z.load(DEFAULT_RUN, "T1") is None


def test_reused_gains_are_not_rewritten():
    out = R._commit_per_zone(DEFAULT_RUN, R.RetuneOptions(), _payload(0.6, 0.98, reused=True, saved=None))
    assert out["search"]["per_zone_gains"]["saved"] is False
    assert Z.load(DEFAULT_RUN, "T1") is None


def test_cache_hit_recommits_after_forget(monkeypatch):
    """FORGET, then the same click: the cached result is served and the six are saved again."""
    stored = _payload(0.6, 0.98, saved=True)
    monkeypatch.setattr(R.cache, "load", lambda kind, key: copy.deepcopy(stored))
    assert Z.load(DEFAULT_RUN, "T1") is None                       # forgotten
    out = R.retune(DEFAULT_RUN)
    assert out["cached"] is True
    assert Z.load(DEFAULT_RUN, "T1")["source_run_hash"] == "case-1"


def test_cache_hit_of_a_restored_case_saves_nothing(monkeypatch):
    stored = _payload(1.2, 0.98, saved=False)
    monkeypatch.setattr(R.cache, "load", lambda kind, key: copy.deepcopy(stored))
    R.retune(DEFAULT_RUN)
    assert Z.load(DEFAULT_RUN, "T1") is None
