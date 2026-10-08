"""The BO-outfeeder summary computes what the report says it does.

    JAX_PLATFORMS=cpu ../.venv/bin/python -m pytest tests/test_aggregate_bo_outfeeder.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE / "src") not in sys.path:
    sys.path.insert(0, str(HERE / "src"))

import aggregate_bo_outfeeder as A  # noqa: E402


def _cell(pool, drift, floor, comm, hgs, cs, ws, b5, b10, floor6=None):
    one = lambda s, n: [{"seed": 0, "S_plant": s, "real_evals": n}]
    three = lambda ss, n: [{"seed": i, "S_plant": s, "real_evals": n} for i, s in enumerate(ss)]
    return {"pool": pool, "drift": drift, "floor": {"S_plant": floor},
            "floor_6d": None if floor6 is None else {"S_plant": floor6},
            "methods": {"commissioned-6": one(comm, 0), "HGS-only": one(hgs, 0),
                        "HGS+BO(5)": one(b5, 5), "HGS+BO(10)": one(b10, 10),
                        "CS-BO(30)": three(cs, 30), "WS-BO(30)": three(ws, 30)}}


CELLS = [
    _cell("P001", "D01", floor=1.0, comm=2.0, hgs=1.10, cs=[1.2, 1.3, 1.4], ws=[1.05, 1.0, 1.1],
          b5=1.10, b10=1.02, floor6=0.8),
    _cell("P001", "D02", floor=2.0, comm=2.0, hgs=2.00, cs=[3.0, 2.2, 2.4], ws=[2.0, 2.0, 2.0],
          b5=2.00, b10=2.00, floor6=1.0),
]


def test_gap_to_floor_per_run():
    s = A.summarise(CELLS)
    hgs = s["arms"]["HGS-only"]
    assert hgs["runs"] == 2 and hgs["real_evals"] == 0
    assert abs(hgs["median_gap_to_floor_pct"] - 5.0) < 1e-9          # 10 % and 0 %


def test_seeded_arms_pool_all_runs():
    cs = A.summarise(CELLS)["arms"]["CS-BO(30)"]
    assert cs["runs"] == 6
    # gaps: 20, 30, 40, 50, 10, 20 -> median 25
    assert abs(cs["median_gap_to_floor_pct"] - 25.0) < 1e-9


def test_change_vs_doing_nothing():
    b10 = A.summarise(CELLS)["arms"]["HGS+BO(10)"]
    # -49 % and 0 % -> median -24.5
    assert abs(b10["median_change_vs_commissioned_pct"] - (-24.5)) < 1e-9


def test_beats_hgs_counts_strict_improvements():
    s = A.summarise(CELLS)["arms"]
    assert s["HGS+BO(10)"]["beats_hgs_only"] == 1          # 1.02 < 1.10; 2.00 == 2.00
    assert s["WS-BO(30)"]["beats_hgs_only"] == 2           # 1.05, 1.0 < 1.10; 1.1 and the 2.0s are ties


def test_price_of_frozen_ends():
    s = A.summarise(CELLS)
    # floor vs 6-D floor: +25 % and +100 % -> median 62.5
    assert abs(s["frozen_ends_price_pct"] - 62.5) < 1e-9


def test_runs_below_the_hgs_floor_are_counted_not_hidden():
    """The floor is HGS on the true plant -- a grid, not the true optimum. A BO run can
    land below it; the summary must say so instead of clipping it to 'at the floor'."""
    cells = [_cell("P001", "D01", floor=1.0, comm=2.0, hgs=1.10, cs=[0.99, 1.3, 1.4],
                   ws=[1.05, 1.0, 1.1], b5=1.10, b10=0.995, floor6=0.8)]
    s = A.summarise(cells)["arms"]
    assert s["CS-BO(30)"]["below_floor"] == 1
    assert s["HGS+BO(10)"]["below_floor"] == 1
    assert s["HGS-only"]["below_floor"] == 0
