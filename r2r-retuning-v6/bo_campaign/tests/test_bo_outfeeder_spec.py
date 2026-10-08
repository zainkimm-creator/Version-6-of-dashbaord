"""The out-feeder BO campaign does exactly what BO-OUTFEEDER-SPEC.md says.

    JAX_PLATFORMS=cpu ../.venv/bin/python -m pytest tests/test_bo_outfeeder_spec.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
for q in (str(HERE.parent), str(HERE / "src")):
    if q not in sys.path:
        sys.path.insert(0, q)

import run_bo_outfeeder as B  # noqa: E402


@pytest.mark.parametrize("arm, budget, warm, split", [
    ("CS-BO(30)", 30, False, (0, 10, 20)),
    ("WS-BO(30)", 30, True, (5, 8, 17)),
    ("HGS+BO(5)", 5, True, (2, 3, 0)),
    ("HGS+BO(10)", 10, True, (5, 3, 2)),
])
def test_seed_random_ei_split(arm, budget, warm, split):
    assert B.bo_split(budget, warm) == split
    assert sum(split) == budget


@pytest.mark.parametrize("arm, budget, warm, split", [
    ("CS-BO(30)", 30, False, (0, 10, 20)),
    ("WS-BO(30)", 30, True, (5, 8, 17)),
    ("HGS+BO(5)", 5, True, (2, 3, 0)),
    ("HGS+BO(10)", 10, True, (5, 3, 2)),
])
def test_split_is_what_skopt_actually_does(arm, budget, warm, split):
    """Run the real _bo on a cheap analytic cost and count what it evaluated."""
    calls = []

    def costs(kp, ti):
        calls.append((float(kp[0][1]), float(ti[0][1])))
        return np.array([(np.log(kp[0][1]) - np.log(40.0)) ** 2 + ((ti[0][1] - 12.0) / 10) ** 2])

    st = B.STRUCTURE
    base = (np.array([10.0, 10.0, 10.0]), np.array([20.0, 20.0, 20.0]))
    warm_x = np.array([60.0, 15.0]) if warm else None
    _, _, _, n, traj = B.bo(costs, st, base, budget, warm_x=warm_x, seed=0)
    assert n == budget == len(calls) == len(traj)
    if warm:
        assert calls[0] == (60.0, 15.0)                    # the star's centre comes first
    rnd = calls[split[0]:split[0] + 3]
    assert [(round(a, 3), round(b, 3)) for a, b in rnd] == list(B.SEED0_RANDOM_DESIGN[:len(rnd)])


def test_star_is_five_axis_points_clipped():
    pts = B.star(np.array([450.0, 25.0]))
    assert len(pts) == 5
    assert [tuple(np.round(p, 6)) for p in pts] == [(450.0, 25.0), (315.0, 25.0), (500.0, 25.0),
                                                     (450.0, 17.5), (450.0, 30.0)]


def test_shared_random_designs_per_seed():
    d = B.random_designs()
    assert d[0][:3] == [(39.817, 25.406), (206.809, 25.494), (48.193, 11.839)]
    assert len({tuple(v[:3]) for v in d.values()}) == 3


def test_gpu_guard_refuses_cpu():
    with pytest.raises(RuntimeError, match="GPU"):
        B.require_gpu(["CpuDevice(id=0)"])
