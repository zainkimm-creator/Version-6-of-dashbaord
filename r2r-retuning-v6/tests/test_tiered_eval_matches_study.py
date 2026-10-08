"""The backend tiered cost and the study's copy must be the same physics.

`backend/validation/retuning_tiered_eval.py` is the canonical implementation;
`multiloop/src/tiers.py` is the copy the pilot campaign ran against. While both
exist they must agree exactly, or the campaign's conclusions describe code the
dashboard does not run. Two copies of a kernel without a gate is how a 400x error
got into a previous study.

Also re-pins T0 against the authors' own evaluator, so the whole tier stack still
rests on arithmetic proven equal to the paper's.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_tiered_eval_matches_study.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from backend.validation import retuning as R
from backend.validation.plants import parameters_for_plant
from backend.validation.retuning_paper_eval import PaperEvaluator
from backend.validation.retuning_tiered_eval import TieredEvaluator as BackendTiered

STUDY_SRC = Path(__file__).resolve().parents[1] / "bo_campaign" / "src"
CASES = [("P001", "D07"), ("P053", "D03"), ("P189", "D01")]
GAINS = [((3.705, 3.705, 3.705), (30.0, 30.0, 30.0)),
         ((12.0, 4.0, 18.0), (9.0, 22.0, 14.5)),
         ((1.5, 60.0, 7.0), (0.75, 5.0, 29.5))]


def _plant(pool, drift):
    base, meta = parameters_for_plant(dict(R.RETUNING_PLANTS)[pool])
    v0 = float(meta.get("v0_mps") or base.feeder_velocity_m_s)
    return R.apply_drift(base, R.DRIFT_BY_CODE[drift]), v0


@pytest.mark.skipif(not (STUDY_SRC / "tiers.py").exists(),
                    reason="study copy not present")
@pytest.mark.parametrize("tier", ["T0", "T1", "T2", "T3"])
def test_backend_matches_the_study_copy(tier):
    if str(STUDY_SRC) not in sys.path:
        sys.path.insert(0, str(STUDY_SRC))
    from tiers import TieredEvaluator as StudyTiered   # noqa: PLC0415

    worst = 0.0
    for pool, drift in CASES:
        params, v0 = _plant(pool, drift)
        a = BackendTiered(params, v0, tier=tier)
        b = StudyTiered(params, v0, tier=tier)
        for kp, ti in GAINS:
            ra = a.evaluate([kp], [ti])[0]
            rb = b.evaluate([kp], [ti])[0]
            worst = max(worst, float(np.max(np.abs(ra - rb))))
    print(f"  {tier}: worst |backend - study| across "
          f"{len(CASES)*len(GAINS)} evaluations = {worst:.3e}")
    assert worst == 0.0, f"{tier} diverged between the two copies by {worst:.3e}"


def test_t0_still_reproduces_the_authors_kernel():
    """The foundation everything else is built on."""
    worst = 0.0
    for pool, drift in CASES:
        params, v0 = _plant(pool, drift)
        paper = PaperEvaluator(params, v0)
        t0 = BackendTiered(params, v0, tier="T0")
        for kp, ti in ((3.705, 30.0), (12.5, 7.25), (100.0, 18.539)):
            a = paper.evaluate([kp], [ti])[0]
            b = t0.evaluate([[kp] * 3], [[ti] * 3])[0]
            worst = max(worst, float(np.max(np.abs(a[:4] - b[:4]))))
    print(f"  T0 vs PaperEvaluator: worst = {worst:.3e}")
    assert worst == 0.0, f"T0 is no longer the authors' arithmetic ({worst:.3e})"


def test_t1_removes_the_settling_cliff():
    """P001/D07: Eq. (12) jumps +72 % across a 1.14 % gain change; T1 must not."""
    params, v0 = _plant("P001", "D07")
    lo, hi = 3.6634, 3.7050
    out = {}
    for tier in ("T0", "T1"):
        ev = BackendTiered(params, v0, tier=tier)
        a = ev.cost([lo] * 3, [30.0] * 3)
        b = ev.cost([hi] * 3, [30.0] * 3)
        out[tier] = 100.0 * (b - a) / a
        print(f"  {tier}: {a:.5f} -> {b:.5f}  [{out[tier]:+.1f} %]")
    assert abs(out["T0"]) > 50.0, "the cliff vanished from T0 -- test no longer meaningful"
    assert abs(out["T1"]) < 10.0, f"T1 still has a cliff ({out['T1']:+.1f} %)"
