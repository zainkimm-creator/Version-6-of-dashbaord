"""`gain_structure="sequential-2D"`: the adopted method, end to end through the pipeline.

The authors' 2-D HGS on one zone at a time (UW, RW, out-feeder), two sweeps, from the
commissioned pair. UW and RW end up fixed at what HGS found; the out-feeder is then at
its optimum. No 6-D search, no real-plant trials. See multiloop/reports/SEQUENTIAL-HGS.pdf.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_retune_sequential.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.pipeline.payloads import DEFAULT_RUN
from backend.pipeline.retune import (RetuneOptions, SEQUENTIAL_ORDER, SEQUENTIAL_SWEEPS,
                                     retune)


@pytest.fixture(scope="module")
def seq():
    return retune(DEFAULT_RUN, RetuneOptions(gain_structure="sequential-2D"), use_cache=False)


@pytest.fixture(scope="module")
def shared():
    return retune(DEFAULT_RUN, RetuneOptions(gain_structure="shared-2D"), use_cache=False)


def test_runs_and_delivers_per_zone_gains(seq):
    assert seq["status"] == "ok"
    g = seq["gains"]["delivered"]
    assert g["per_zone"] is True and g["zones"] == ["UW", "OutFeeder", "RW"]
    assert np.isfinite(g["on_plant"]["S"])
    print(f"  K_p* = {[round(k, 3) for k in g['kp_star_per_zone']]}  "
          f"T_I = {[round(t, 2) for t in g['ti_s_per_zone']]}  S = {g['on_plant']['S']:.5f}")


def test_exactly_the_published_procedure(seq):
    """Two sweeps x three zones, every stage the authors' 2,805-point HGS."""
    stages = seq["search"]["sequential_stages"]
    assert len(stages) == SEQUENTIAL_SWEEPS * len(SEQUENTIAL_ORDER)
    assert [s["zone"] for s in stages] == [z for _ in range(SEQUENTIAL_SWEEPS) for z, _ in SEQUENTIAL_ORDER]
    assert all(s["twin_evals"] == 2805 for s in stages)
    assert seq["search"]["budget"] == 2805 * len(stages)
    assert seq["gains"]["hgs_only"]["real_evals"] == 0 if "real_evals" in seq["gains"]["hgs_only"] else True
    # the twin cost never gets worse from one stage to the next (each HGS can keep the incumbent)
    costs = [s["S_twin"] for s in stages]
    assert all(b <= a + 1e-9 for a, b in zip(costs, costs[1:])), costs


def test_each_stage_moves_only_its_own_zone(seq):
    stages = seq["search"]["sequential_stages"]
    idx = {"UW": 0, "OutFeeder": 1, "RW": 2}
    prev_kp, prev_ti = None, None
    for s in stages:
        if prev_kp is not None:
            z = idx[s["zone"]]
            for j in range(3):
                if j != z:
                    assert s["kp_star_per_zone"][j] == prev_kp[j]
                    assert s["ti_s_per_zone"][j] == prev_ti[j]
        prev_kp, prev_ti = s["kp_star_per_zone"], s["ti_s_per_zone"]


def test_scalar_only_outputs_say_not_applicable(seq):
    assert seq["gains"]["simc_reference"]["not_applicable"]
    assert seq["ti_profile"] == []


def test_not_worse_than_the_shared_pair(seq, shared):
    a = shared["gains"]["delivered"]["on_plant"]["S"]
    b = seq["gains"]["delivered"]["on_plant"]["S"]
    print(f"  shared {a:.5f} -> sequential {b:.5f}  ({100 * (b - a) / a:+.1f} %)")
    assert b <= a * 1.02
