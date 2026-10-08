"""RetuneOptions.cost_tier selects the scoring cost. T0 must be unchanged.

Since 2026-10-07 the default is "T1", the repaired cost (user's decision). T0, the
authors' Eq. (12), stays selectable, and a T0 scorer must still be bit-identical
to the PaperEvaluator the pipeline used before tiers existed, while T1-T3 must be
reachable, finite, and actually different.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_retune_cost_tier.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.pipeline.retune import DEFAULT_COST_TIER, RetuneOptions, _Scorer
from backend.validation import retuning as R
from backend.validation.plants import parameters_for_plant
from backend.validation.retuning_paper_eval import PaperEvaluator

GAINS = [(3.705, 30.0), (12.5, 7.25), (100.0, 18.539)]


@pytest.fixture(scope="module")
def plant():
    base, meta = parameters_for_plant(dict(R.RETUNING_PLANTS)["P001"])
    v0 = float(meta.get("v0_mps") or base.feeder_velocity_m_s)
    return R.apply_drift(base, R.DRIFT_BY_CODE["D07"]), v0


def test_the_default_is_the_repaired_cost():
    assert DEFAULT_COST_TIER == "T1"
    assert RetuneOptions().cost_tier == "T1"


def test_t0_scorer_is_bit_identical_to_the_paper_evaluator(plant):
    """The no-regression promise, on all three call paths."""
    params, v0 = plant
    scorer = _Scorer(params, v0, "T0")
    paper = PaperEvaluator(params, v0)

    kps = [k for k, _ in GAINS]
    tis = [t for _, t in GAINS]
    worst_batch = float(np.max(np.abs(np.asarray(scorer.batch(kps, tis))
                                      - np.asarray(paper.costs(kps, tis)))))
    worst_scalar = max(abs(scorer.scalar(k, t) - paper.cost(k, t)) for k, t in GAINS)
    b_s, b_p = scorer.breakdown(*GAINS[0]), paper.breakdown(*GAINS[0])
    worst_break = max(abs(b_s[k] - b_p[k]) for k in b_p if b_p[k] is not None)

    print(f"  batch {worst_batch:.3e}  scalar {worst_scalar:.3e}  "
          f"breakdown {worst_break:.3e}")
    assert worst_batch == 0.0 and worst_scalar == 0.0 and worst_break == 0.0


# Gains a real line would run. The high-gain member of GAINS (kp = 100) is stable
# only because Eq. (12) models no drive; see the test below.
SANE_GAINS = [(3.705, 30.0), (12.5, 7.25), (8.0, 4.0)]


@pytest.mark.parametrize("tier", ["T1", "T2", "T3"])
def test_repaired_tiers_are_reachable_finite_and_different(plant, tier):
    params, v0 = plant
    t0 = _Scorer(params, v0, "T0")
    ts = _Scorer(params, v0, tier)
    for kp, ti in SANE_GAINS:
        a, b = t0.scalar(kp, ti), ts.scalar(kp, ti)
        assert np.isfinite(b), f"{tier} not finite at kp={kp}, ti={ti}"
        assert a != b, f"{tier} returned the T0 value -- tier not wired through"
    assert ts.breakdown(*SANE_GAINS[0])["finite"] is True


def test_idealised_cost_calls_an_unstable_gain_merely_mediocre(plant):
    """Not a bug -- the point of T2.

    K_p* = 100 scores a finite (poor) cost under Eq. (12), which models no drive at
    all. Add a torque limit, a slew limit and a real feedback path and the same gain
    runs away. A cost that cannot tell "bad" from "unstable" cannot be commissioned
    against, and this is the sharpest single illustration of it.
    """
    params, v0 = plant
    kp, ti = 100.0, 18.539
    s_t0 = _Scorer(params, v0, "T0").scalar(kp, ti)
    s_t2 = _Scorer(params, v0, "T2").scalar(kp, ti)
    print(f"  K_p*={kp}: Eq.(12) S = {s_t0:.3f} (finite)   with a real drive: {s_t2}")
    assert np.isfinite(s_t0), "T0 should still call this merely bad"
    assert not np.isfinite(s_t2), "T2 no longer detects this as unstable"


def test_a_bad_tier_is_rejected_not_silently_ignored(plant):
    params, v0 = plant
    with pytest.raises(ValueError):
        _Scorer(params, v0, "T9")
    with pytest.raises(ValueError):
        RetuneOptions(cost_tier="paper")


def test_the_tier_is_part_of_the_cache_key():
    """Two tiers must not share a cached result -- that is how stale physics ships."""
    from backend.pipeline.retune import _retune_key
    from backend.pipeline.payloads import DEFAULT_RUN

    run = DEFAULT_RUN
    keys = {t: _retune_key(run, RetuneOptions(cost_tier=t))
            for t in ("T0", "T1", "T2", "T3")}
    print("  distinct keys:", len(set(keys.values())), "of", len(keys))
    assert len(set(keys.values())) == len(keys), f"cache key collision: {keys}"
