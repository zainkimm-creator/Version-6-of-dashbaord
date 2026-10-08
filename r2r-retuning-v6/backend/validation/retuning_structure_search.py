"""A dimension-generic search, so every structure is searched the same way.

The paper's HGS is a 30x30 grid plus an LHS fill plus a local grid plus a polish:
2,805 evaluations over two dimensions. That construction does not survive a change
of dimension -- 30^6 is 7.3e8 points -- so the grid stages become space-filling
samples that shrink around the incumbent. Coverage, then zoom, then polish.

Crucially the SAME routine searches every structure, with the sample budget scaled
to dimension (about 1,500 points per dimension in the global stage). If the 2-D and
6-D arms used different searches, a difference between them would be a difference in
search effort, not in parameterisation, and the whole comparison would be void.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NONFINITE = 1e3
ZOOM_FRACTION = 0.25
GLOBAL_PER_DIM = 1500
ZOOM1_FRAC, ZOOM2_FRAC = 0.25, 0.10
POLISH_KP = (0.97, 0.99, 1.01, 1.03)
POLISH_TI = (-0.5, -0.15, 0.15, 0.5)


@dataclass
class SearchResult:
    x: np.ndarray
    kp: np.ndarray
    ti: np.ndarray
    cost: float
    evaluations: int
    stage_counts: dict = field(default_factory=dict)
    on_bound: list = field(default_factory=list)


def _lhs(n: int, d: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    return qmc.LatinHypercube(d=d, seed=seed).random(n)


def _sample(n, d, seed, lo, hi, log):
    u = _lhs(n, d, seed)
    out = np.empty((n, d))
    for j in range(d):
        if log[j]:
            out[:, j] = np.exp(np.log(lo[j]) + u[:, j] * (np.log(hi[j]) - np.log(lo[j])))
        else:
            out[:, j] = lo[j] + u[:, j] * (hi[j] - lo[j])
    return out


def _window(centre, lo, hi, frac, log):
    d = centre.size
    wlo, whi = np.empty(d), np.empty(d)
    for j in range(d):
        if log[j]:
            span = frac * (np.log(hi[j]) - np.log(lo[j]))
            wlo[j] = np.clip(np.exp(np.log(centre[j]) - span), lo[j], hi[j])
            whi[j] = np.clip(np.exp(np.log(centre[j]) + span), lo[j], hi[j])
        else:
            span = frac * (hi[j] - lo[j])
            wlo[j] = np.clip(centre[j] - span, lo[j], hi[j])
            whi[j] = np.clip(centre[j] + span, lo[j], hi[j])
    return wlo, whi


def search(batch_costs, structure, *, baseline=None, seed: int = 0, scale: float = 1.0):
    """`batch_costs(kp[n,3], ti[n,3]) -> cost[n]`. Deterministic for a given seed.

    `scale` shrinks every stage uniformly (a pilot runs at scale < 1); it multiplies
    all structures equally, so the comparison between them is unaffected.
    """
    d = structure.dim
    lo = np.array([b[0] for b in structure.bounds], dtype=float)
    hi = np.array([b[1] for b in structure.bounds], dtype=float)
    log = [bool(b[2]) for b in structure.bounds]

    stages, total = {}, 0
    best_x, best_kp, best_ti, best_cost = None, None, None, np.inf

    def score(x, name):
        nonlocal best_x, best_kp, best_ti, best_cost, total
        kp, ti = structure.expand_batch(x, baseline)
        c = np.asarray(batch_costs(kp, ti), dtype=float)
        c = np.where(np.isfinite(c), c, NONFINITE)
        i = int(np.argmin(c))
        if c[i] < best_cost:
            best_cost, best_x = float(c[i]), x[i].copy()
            best_kp, best_ti = kp[i].copy(), ti[i].copy()
        stages[name] = int(x.shape[0])
        total += int(x.shape[0])

    n_global = max(200, int(GLOBAL_PER_DIM * d * scale))
    score(_sample(n_global, d, seed, lo, hi, log), "global")

    for k, (name, frac) in enumerate((("zoom1", ZOOM1_FRAC), ("zoom2", ZOOM2_FRAC)), 1):
        n = max(100, int(n_global * frac))
        wlo, whi = _window(best_x, lo, hi, ZOOM_FRACTION ** k, log)
        score(_sample(n, d, seed + 101 * k, wlo, whi, log), name)

    # pattern polish: one coordinate at a time, both directions
    pts = [best_x.copy()]
    for j in range(d):
        steps = POLISH_KP if log[j] else POLISH_TI
        for s in steps:
            y = best_x.copy()
            y[j] = np.clip(y[j] * s if log[j] else y[j] + s, lo[j], hi[j])
            pts.append(y)
    score(np.asarray(pts), "polish")

    tol = 1e-9
    on_bound = ["lower" if abs(best_x[j] - lo[j]) < tol else
                ("upper" if abs(best_x[j] - hi[j]) < tol else None) for j in range(d)]
    return SearchResult(best_x, best_kp, best_ti, best_cost, total, stages, on_bound)
