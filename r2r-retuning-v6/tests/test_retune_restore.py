"""Step 7 "Validate and restore": the twin's gains reach the line only if they beat
today's gains there; otherwise the commissioned gains are restored.

The restore is derived from two numbers every retune payload already holds -- the
candidate's S on the drifted plant and the commissioned gains' S on the same plant
-- so it is applied to cached and fresh results alike, and the retune cache
(and the precompute feeding it) stays valid.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_retune_restore.py -q
"""

from __future__ import annotations

import copy

from backend.pipeline import retune as R
from backend.pipeline.payloads import DEFAULT_RUN


def _payload(s_candidate, s_commissioned, c_target=0.2):
    def block(kp, ti, s, **extra):
        return {"kp_star": None, "ti_s": None, "kp_star_per_zone": kp, "ti_s_per_zone": ti,
                "per_zone": True, "on_plant": {"S": s}, **extra}

    commissioned = block([4.0, 4.0, 4.0], [30.0, 30.0, 30.0], s_commissioned)
    candidate = block([9.0, 3.0, 9.0], [10.0, 20.0, 10.0], s_candidate, S_twin=0.1,
                      source="recommended (HGS-only, twin optimum as searched)")
    return {
        "status": "ok",
        "steps": [{"step": 5, "name": "Build digital twin and validate", "status": "pass"},
                  {"step": 4, "name": "Run system identification", "status": "pass"},
                  {"step": 6, "name": "Retune on the digital twin", "status": "pass"},
                  {"step": 7, "name": "Validate and restore", "status": "fail", "detail": "old"}],
        "gains": {"delivered": candidate, "recommended": copy.deepcopy(candidate),
                  "commissioned": commissioned, "reference_on_plant": block([1, 1, 1], [1, 1, 1], 0.05)},
        "acceptance": {"S_delivered": s_candidate, "C_target": c_target, "margin": 1.0,
                       "passed": s_candidate is not None and s_candidate <= c_target},
        "ratio_vs_reference": None if s_candidate is None else s_candidate / 0.05,
    }


def _step7(p):
    return next(s for s in p["steps"] if s["step"] == 7)


def test_worse_candidate_restores_the_commissioned_gains():
    p = R.apply_restore(_payload(s_candidate=3.33, s_commissioned=3.07))
    d = p["gains"]["delivered"]
    assert d["kp_star_per_zone"] == [4.0, 4.0, 4.0] and d["ti_s_per_zone"] == [30.0, 30.0, 30.0]
    assert d["on_plant"]["S"] == 3.07
    assert p["restore"]["restored"] is True
    assert p["gains"]["twin_candidate"]["kp_star_per_zone"] == [9.0, 3.0, 9.0]
    assert _step7(p)["status"] == "restored"
    assert p["acceptance"]["S_delivered"] == 3.07
    assert abs(p["ratio_vs_reference"] - 3.07 / 0.05) < 1e-12


def test_better_candidate_is_delivered_even_above_c_target():
    """The P06 case: the retune cuts S 0.322 -> 0.245 but cannot reach the pre-drift 0.198.
    Restoring would make the line worse, so the candidate stays."""
    p = R.apply_restore(_payload(s_candidate=0.245, s_commissioned=0.322, c_target=0.198))
    assert p["restore"]["restored"] is False
    assert p["gains"]["delivered"]["kp_star_per_zone"] == [9.0, 3.0, 9.0]
    assert p["acceptance"]["passed"] is False          # C_target still reported honestly
    assert _step7(p)["status"] == "fail"


def test_tie_keeps_the_gains_already_running():
    p = R.apply_restore(_payload(s_candidate=2.0, s_commissioned=2.0))
    assert p["restore"]["restored"] is True


def test_diverging_candidate_is_restored():
    p = R.apply_restore(_payload(s_candidate=None, s_commissioned=0.5))
    assert p["restore"]["restored"] is True
    assert p["gains"]["delivered"]["on_plant"]["S"] == 0.5


def test_passing_candidate_passes():
    p = R.apply_restore(_payload(s_candidate=0.15, s_commissioned=0.3, c_target=0.2))
    assert p["restore"]["restored"] is False and _step7(p)["status"] == "pass"


def test_idempotent_and_does_not_mutate_input():
    raw = _payload(s_candidate=3.33, s_commissioned=3.07)
    before = copy.deepcopy(raw)
    once = R.apply_restore(raw)
    assert raw == before
    assert R.apply_restore(once) == once


def test_failed_identification_passes_through():
    p = {"status": "identification_failed", "gains": None, "steps": []}
    assert R.apply_restore(p) == p


def test_cache_hit_gets_the_restore(monkeypatch):
    """A payload cached before the restore existed is restored on the way out."""
    stored = _payload(s_candidate=3.33, s_commissioned=3.07)
    monkeypatch.setattr(R.cache, "load", lambda kind, key: copy.deepcopy(stored))
    out = R.retune(DEFAULT_RUN)
    assert out["cached"] is True and out["restore"]["restored"] is True
    assert out["gains"]["delivered"]["kp_star_per_zone"] == [4.0, 4.0, 4.0]


def test_assumption_is_documented():
    assert any(a["id"] == "A9" and "restore" in a["what"].lower() for a in R.ASSUMPTIONS)
