"""The Section 4.2 search protocol as the authors run it (reply Q1b, Q3, Q4).

* Box: K_p* in [1, 500] with a log-uniform prior; T_I in [0.5, 30] s ABSOLUTE
  with skopt's default uniform prior.
* gp_minimize(objective, space, n_calls=budget_remaining,
  n_initial_points=max(3, min(10, budget_remaining // 3)), x0, y0,
  acq_func="EI", random_state=seed, noise=1e-6), default GP, raw cost.
* Warm-started arms pass a deterministic five-point axis-aligned star (centre,
  K_p* -/+30 %, T_I -/+30 %, clipped to the box), truncated to at most half the
  budget, as x0 with the evaluated costs as y0; the remaining budget is
  max(3, budget - len(x0)). skopt 0.10.2 draws n_initial_points random points on
  top of x0, so the seed/random/EI split is CS-BO 0/10/20, WS-BO 5/8/17,
  HGS+BO(5) 2/3/0 and HGS+BO(10) 5/3/2.
* WS-BO centre: (100, clip(auto_Ti(theta_hat), 0.5, 30)). HGS+BO centre: the
  twin optimum. CS/WS-BO seeds {0, 1, 2}; the HGS family is deterministic, seed 0.
* The reported best is the running minimum including the star evaluations.

The hierarchical grid search is the one piece the reply does not re-specify;
its stage split and grid spacing are this module's documented choice
(`HGS_STAGES`), searched over the same box the BO arms use.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

KP_BOUNDS = (1.0, 500.0)
TI_BOUNDS_S = (0.5, 30.0)
GP_NOISE = 1e-6
STAR_STEP = 0.30
WS_KP_CENTRE = 100.0
# Coarse 30x30 grid, 1,000-point Latin hypercube, 30x30 fine grid around the
# incumbent, 5-point local star polish: 2,805 twin evaluations, as S8.1 prints.
HGS_STAGES = {"coarse_side": 30, "lhs": 1000, "fine_side": 30, "polish": 5}
# A diverging closed loop returns no cost. The GP cannot take inf; this finite
# stand-in is recorded per run (`nonfinite_evaluations`) wherever it is used.
NONFINITE_COST = 1.0e3


def space():
    from skopt.space import Real

    return [Real(*KP_BOUNDS, prior="log-uniform", name="kp_star"),
            Real(*TI_BOUNDS_S, name="ti_s")]


def clip_point(kp: float, ti: float) -> tuple[float, float]:
    return (min(max(kp, KP_BOUNDS[0]), KP_BOUNDS[1]), min(max(ti, TI_BOUNDS_S[0]), TI_BOUNDS_S[1]))


def star(kp: float, ti: float, max_points: int = 5) -> list[tuple[float, float]]:
    """Centre, K_p* -30 %, K_p* +30 %, T_I -30 %, T_I +30 %, clipped; truncated."""

    points = [(kp, ti), (kp * (1 - STAR_STEP), ti), (kp * (1 + STAR_STEP), ti),
              (kp, ti * (1 - STAR_STEP)), (kp, ti * (1 + STAR_STEP))]
    return [clip_point(*p) for p in points][:max(0, max_points)]


def n_initial_points(remaining: int) -> int:
    return max(3, min(10, remaining // 3))


@dataclass
class BORun:
    points: list[tuple[float, float]]
    costs: list[float]
    split: dict[str, int]
    nonfinite_evaluations: int = 0
    running_best: list[float] = field(default_factory=list)

    @property
    def best_index(self) -> int:
        return min(range(len(self.costs)), key=self.costs.__getitem__)


def run_bo(cost: Callable[[float, float], float], *, budget: int, seed: int,
           warm_start: Sequence[tuple[float, float]] | None = None) -> BORun:
    """One BO arm, exactly the authors' gp_minimize call."""

    from skopt import gp_minimize

    x0 = [list(p) for p in (warm_start or [])][: budget // 2]
    nonfinite = 0

    def objective(x):
        nonlocal nonfinite
        value = cost(float(x[0]), float(x[1]))
        if not math.isfinite(value):
            nonfinite += 1
            return NONFINITE_COST
        return float(value)

    y0 = [objective(p) for p in x0]
    remaining = max(3, budget - len(x0))
    n_init = n_initial_points(remaining)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = gp_minimize(
            objective, space(), n_calls=remaining, n_initial_points=n_init,
            x0=x0 or None, y0=y0 or None, acq_func="EI", random_state=seed, noise=GP_NOISE,
        )
    points = [(float(a), float(b)) for a, b in result.x_iters]
    costs = [float(v) for v in result.func_vals]
    running, best = [], math.inf
    for c in costs:
        best = min(best, c)
        running.append(best)
    return BORun(points, costs,
                 {"seed_points": len(x0), "random": n_init, "ei": remaining - n_init},
                 nonfinite, running)


@dataclass
class HGSResult:
    kp: float
    ti: float
    cost: float
    evaluations: int
    stage_counts: dict[str, int]
    kp_on_bound: str | None
    ti_on_bound: str | None


def _bound(value: float, bounds: tuple[float, float]) -> str | None:
    if math.isclose(value, bounds[0], rel_tol=1e-9):
        return "lower"
    if math.isclose(value, bounds[1], rel_tol=1e-9):
        return "upper"
    return None


def hierarchical_grid_search(batch_costs: Callable[[np.ndarray, np.ndarray], np.ndarray],
                             *, seed: int = 0) -> HGSResult:
    """Deterministic 2,805-point search over the authors' box (see HGS_STAGES)."""

    from scipy.stats import qmc

    kp_lo, kp_hi = KP_BOUNDS
    ti_lo, ti_hi = TI_BOUNDS_S
    stages: dict[str, int] = {}
    kps: list[np.ndarray] = []
    tis: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    def score(kp, ti, name):
        kp = np.asarray(kp, dtype=float).ravel()
        ti = np.asarray(ti, dtype=float).ravel()
        kps.append(kp); tis.append(ti); vals.append(np.asarray(batch_costs(kp, ti), dtype=float))
        stages[name] = kp.size

    def incumbent():
        k, t, v = np.concatenate(kps), np.concatenate(tis), np.concatenate(vals)
        i = int(np.argmin(np.where(np.isfinite(v), v, np.inf)))
        return float(k[i]), float(t[i]), float(v[i])

    n = HGS_STAGES["coarse_side"]
    g_kp, g_ti = np.meshgrid(np.geomspace(kp_lo, kp_hi, n), np.linspace(ti_lo, ti_hi, n), indexing="ij")
    score(g_kp, g_ti, "coarse")

    u = qmc.LatinHypercube(d=2, seed=seed).random(HGS_STAGES["lhs"])
    score(np.exp(np.log(kp_lo) + u[:, 0] * (np.log(kp_hi) - np.log(kp_lo))),
          ti_lo + u[:, 1] * (ti_hi - ti_lo), "lhs")

    kp0, ti0, _ = incumbent()
    n = HGS_STAGES["fine_side"]
    half_ti = 0.125 * (ti_hi - ti_lo)
    g_kp, g_ti = np.meshgrid(np.geomspace(max(kp_lo, kp0 / 2), min(kp_hi, kp0 * 2), n),
                             np.linspace(max(ti_lo, ti0 - half_ti), min(ti_hi, ti0 + half_ti), n),
                             indexing="ij")
    score(g_kp, g_ti, "fine")

    kp0, ti0, _ = incumbent()
    local = [clip_point(kp0 * a, ti0 * b) for a, b in ((1.05, 1), (0.95, 1), (1, 1.05), (1, 0.95), (1.02, 1.02))]
    score([p[0] for p in local], [p[1] for p in local], "polish")

    kp0, ti0, best = incumbent()
    return HGSResult(kp0, ti0, best, int(sum(stages.values())), stages,
                     _bound(kp0, KP_BOUNDS), _bound(ti0, TI_BOUNDS_S))
