"""Out-feeder BO campaign, on the GPU: the paper's BO methods for the adopted two-stage gains.

Spec: reports/BO-OUTFEEDER-SPEC.md. Plan: docs/plans/2026-10-07-t1-two-stage-bo.md (Task 1).

Two-stage gains (user, 2026-10-07): a free (K_p*, T_I) per zone is found ONCE per plant and
saved; every later case freezes UW and RW at those saved values and retunes only the
out-feeder. Here "once" is commissioning: the 6-D twin search on the PRE-DRIFT plant's twin.

Per cell (6 plants x 10 drifts), cost tier T1, scored on the TRUE drifted plant:
    commissioned-6   the saved six gains, no retune                      0 real evals
    HGS-only         authors' 2-D HGS on the twin, out-feeder only        0
    HGS+BO(5)        BO on the plant, warm star around the HGS answer     5
    HGS+BO(10)       same                                                 10
    CS-BO(30)        BO on the plant, cold                                30   seeds 0,1,2
    WS-BO(30)        BO on the plant, warm star around the SysID point    30   seeds 0,1,2
                     (K_p* = 100, T_I from theta-hat, clipped to the box)
    floor            HGS on the true drifted plant, ends frozen           (reference)
The 6-D floor (free six gains on the true plant) is read from the 2026-10-06 paper-methods
cells (same plant, drift, tier; deterministic, no identification).

    python src/run_bo_outfeeder.py --commission           # 6 plants, once
    python src/run_bo_outfeeder.py --shard 0/4             # one GPU worker of four
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
for q in (str(REPO), str(HERE / "src")):
    if q not in sys.path:
        sys.path.insert(0, q)

from backend.validation.retuning_structures import BY_NAME  # noqa: E402

OUT = HERE / "reports" / "bo_outfeeder_cells"
COMMISSION = OUT / "commissioning"
PAPER_CELLS = HERE / "reports" / "paper_cells"
TIER = "T1"
PLANTS = ("P001", "P049", "P053", "P158", "P186", "P189")
DRIFTS = tuple(f"D{i:02d}" for i in range(1, 11))
STRUCTURE = BY_NAME["outfeeder-only-2D"]
PER_ZONE = BY_NAME["full-6D"]
SYSID_KP = 100.0
STAR_STEP = 0.30
BO_SEEDS = (0, 1, 2)
# skopt 0.10.2, random_state=0, K_p* log-uniform [1, 500], T_I uniform [0.5, 30] s
SEED0_RANDOM_DESIGN = ((39.817, 25.406), (206.809, 25.494), (48.193, 11.839))


# ----------------------------------------------------------------------------- #
# the BO arithmetic (spec section 2)
# ----------------------------------------------------------------------------- #
def bo_split(budget: int, warm: bool) -> tuple[int, int, int]:
    """(star seeds, random initial points, EI steps) -- what gp_minimize will do."""
    seeds = min(budget // 2, 5) if warm else 0
    remaining = max(3, budget - seeds)
    rnd = min(max(3, min(10, remaining // 3)), remaining)
    return seeds, rnd, remaining - rnd


def star(x: np.ndarray, limit: int | None = None) -> list[np.ndarray]:
    """Centre, K_p* -/+30 %, T_I -/+30 %, each clipped to the box."""
    lo = np.array([b[0] for b in STRUCTURE.bounds]); hi = np.array([b[1] for b in STRUCTURE.bounds])
    x = np.asarray(x, dtype=float)
    pts = [np.clip(x.copy(), lo, hi)]
    for j in range(2):
        for f in (1 - STAR_STEP, 1 + STAR_STEP):
            y = np.clip(x.copy(), lo, hi); y[j] = float(np.clip(x[j] * f, lo[j], hi[j])); pts.append(y)
    return pts[:limit] if limit else pts


def _space():
    from skopt.space import Real
    return [Real(b[0], b[1], prior="log-uniform" if b[2] else "uniform", name=n)
            for b, n in zip(STRUCTURE.bounds, ("kp_star", "ti_s"))]


def bo(plant_costs, st, baseline, budget, warm_x=None, seed=0):
    """-> (kp3, ti3, best cost, real evals, trajectory). The out-feeder's (K_p*, T_I) only."""
    from skopt import gp_minimize

    trajectory = []

    def cost_of_x(x):
        kp, ti = st.expand_batch(np.asarray(x, dtype=float)[None, :], baseline)
        c = float(plant_costs(kp, ti)[0])
        c = c if np.isfinite(c) else 1e3
        trajectory.append({"kp_star": float(x[0]), "ti_s": float(x[1]), "S": c})
        return c

    n_seed, n_rnd, _ = bo_split(budget, warm_x is not None)
    x0 = y0 = None
    if warm_x is not None:
        x0 = [[float(v) for v in p] for p in star(warm_x, limit=n_seed)]
        y0 = [cost_of_x(p) for p in x0]
    remaining = max(3, budget - (len(x0) if x0 else 0))
    res = gp_minimize(cost_of_x, _space(), n_calls=remaining, n_initial_points=n_rnd,
                      x0=x0, y0=y0, acq_func="EI", noise=1e-6, random_state=seed)
    kp, ti = st.expand_batch(np.asarray(res.x, dtype=float)[None, :], baseline)
    return kp[0], ti[0], float(res.fun), len(trajectory), trajectory


def random_designs(n: int = 3) -> dict[int, list[tuple[float, float]]]:
    """The first n random initial points skopt draws for each run seed."""
    from skopt import Optimizer
    out = {}
    for seed in BO_SEEDS:
        opt = Optimizer(_space(), base_estimator="GP", n_initial_points=10, acq_func="EI",
                        random_state=seed)
        pts = []
        for _ in range(n):
            x = opt.ask(); pts.append((round(float(x[0]), 3), round(float(x[1]), 3))); opt.tell(x, 0.0)
        out[seed] = pts
    return out


def require_gpu(devices) -> None:
    names = [str(d) for d in devices]
    if not any(("cuda" in s.lower()) or ("gpu" in s.lower()) for s in names):
        raise RuntimeError(f"GPU required, JAX sees only {names}: export the venv's nvidia lib dirs "
                           "on LD_LIBRARY_PATH (see start_dashboard.sh)")


# ----------------------------------------------------------------------------- #
# the campaign
# ----------------------------------------------------------------------------- #
def _world(pool, drift=None):
    from backend.validation import retuning as R
    from backend.validation.plants import parameters_for_plant
    from backend.validation.retuning_tiered_eval import TieredEvaluator
    base, meta = parameters_for_plant(dict(R.RETUNING_PLANTS)[pool])
    v0 = float(meta.get("v0_mps") or base.feeder_velocity_m_s)
    plant = base if drift is None else R.apply_drift(base, R.DRIFT_BY_CODE[drift])
    twin, fit = R.identify_twin(plant, R.PROTOCOLS["field_matched"], meta)
    return R, v0, plant, twin, fit, TieredEvaluator


def commission(pool) -> dict:
    """The saved six gains: 6-D twin search on the pre-drift plant's twin, seed 0."""
    from run_paper_methods import _twin_search
    path = COMMISSION / f"{pool}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    R, v0, plant, twin, fit, TE = _world(pool)
    tw, pl = TE(twin, v0, tier=TIER), TE(plant, v0, tier=TIER)
    x, kp, ti, s_twin, n, how = _twin_search(tw.costs, PER_ZONE, None, seed=0)
    rec = {"pool": pool, "tier": TIER, "kp": kp.tolist(), "ti": ti.tolist(), "S_twin": s_twin,
           "S_plant_predrift": float(pl.costs(kp[None, :], ti[None, :])[0]),
           "twin_evals": n, "search": how, "twin_mare_percent": float(fit["mare_theta_percent"]),
           "seconds": time.time() - t0}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(rec, indent=1)); os.replace(tmp, path)
    return rec


def run_cell(pool, drift) -> dict:
    from run_paper_methods import _twin_search
    t0 = time.time()
    comm = json.loads((COMMISSION / f"{pool}.json").read_text())
    baseline = (np.array(comm["kp"], dtype=float), np.array(comm["ti"], dtype=float))
    R, v0, plant, twin, fit, TE = _world(pool, drift)
    tw, pl = TE(twin, v0, tier=TIER), TE(plant, v0, tier=TIER)
    on_plant = lambda kp, ti: float(pl.costs(np.asarray(kp)[None, :], np.asarray(ti)[None, :])[0])

    ti_hat = float(R.plant_auto_ti_s(twin, v0))
    op_raw = np.array([SYSID_KP, ti_hat])
    lo = np.array([b[0] for b in STRUCTURE.bounds]); hi = np.array([b[1] for b in STRUCTURE.bounds])
    op_x = np.clip(op_raw, lo, hi)

    m = {}
    m["commissioned-6"] = [{"seed": 0, "kp": comm["kp"], "ti": comm["ti"], "real_evals": 0,
                            "S_plant": on_plant(baseline[0], baseline[1])}]
    x, kp, ti, s_twin, n_tw, how = _twin_search(tw.costs, STRUCTURE, baseline, seed=0)
    s_hgs = on_plant(kp, ti)
    m["HGS-only"] = [{"seed": 0, "kp": kp.tolist(), "ti": ti.tolist(), "S_twin": s_twin,
                      "S_plant": s_hgs, "real_evals": 0, "twin_evals": n_tw, "search": how}]
    for b in (5, 10):
        bkp, bti, c, n, traj = bo(pl.costs, STRUCTURE, baseline, b, warm_x=x, seed=0)
        m[f"HGS+BO({b})"] = [{"seed": 0, "kp": bkp.tolist(), "ti": bti.tolist(),
                              "S_plant": min(c, s_hgs), "S_plant_raw": c, "real_evals": n,
                              "improved_on_hgs": bool(c < s_hgs), "split": bo_split(b, True),
                              "trajectory": traj}]
    m["CS-BO(30)"], m["WS-BO(30)"] = [], []
    for seed in BO_SEEDS:
        ckp, cti, c, n, traj = bo(pl.costs, STRUCTURE, baseline, 30, warm_x=None, seed=seed)
        m["CS-BO(30)"].append({"seed": seed, "kp": ckp.tolist(), "ti": cti.tolist(), "S_plant": c,
                               "real_evals": n, "split": bo_split(30, False), "trajectory": traj})
        wkp, wti, c, n, traj = bo(pl.costs, STRUCTURE, baseline, 30, warm_x=op_x, seed=seed)
        m["WS-BO(30)"].append({"seed": seed, "kp": wkp.tolist(), "ti": wti.tolist(), "S_plant": c,
                               "real_evals": n, "split": bo_split(30, True), "trajectory": traj,
                               "warm_start": {"kp_star": SYSID_KP, "ti_from_theta_hat_s": ti_hat,
                                              "ti_used_s": float(op_x[1]),
                                              "clipped_to_box": bool(op_x[1] != ti_hat)}})
    fx, fkp, fti, fcost, fn, fhow = _twin_search(pl.costs, STRUCTURE, baseline, seed=7)
    ref = PAPER_CELLS / f"{pool}__{drift}__perzone__s0.json"
    floor6 = json.loads(ref.read_text())["floor"] if ref.exists() else None
    import jax
    return {"pool": pool, "drift": drift, "tier": TIER, "structure": STRUCTURE.name,
            "commissioning": comm, "twin_mare_percent": float(fit["mare_theta_percent"]),
            "methods": m,
            "floor": {"kp": fkp.tolist(), "ti": fti.tolist(), "S_plant": fcost, "search": fhow},
            "floor_6d": floor6, "device": str(jax.devices()[0]), "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commission", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default=None, help="POOL/DRIFT, for a probe")
    ap.add_argument("--plants", default=None, help="comma list: only these plants' cells")
    ap.add_argument("--reverse", action="store_true",
                    help="walk the cells from the end (helpers meeting forward workers)")
    a = ap.parse_args()
    import jax
    require_gpu(jax.devices())
    OUT.mkdir(parents=True, exist_ok=True)
    if a.commission:
        for p in PLANTS:
            r = commission(p)
            print(json.dumps({"commissioned": p, "S_twin": r["S_twin"], "seconds": r.get("seconds")}), flush=True)
        return
    i, n = (int(v) for v in a.shard.split("/"))
    cells = [(p, d) for p in PLANTS for d in DRIFTS]
    if a.only:
        cells = [tuple(a.only.split("/"))]
    if a.plants:
        keep = set(a.plants.split(","))
        cells = [c for c in cells if c[0] in keep]
    order = list(enumerate(cells))
    if a.reverse:
        order.reverse()
    for k, (p, d) in order:
        if not a.only and k % n != i:
            continue
        path = OUT / f"{p}__{d}.json"
        if path.exists():
            print(json.dumps({"cell": f"{p}/{d}", "status": "cached"}), flush=True)
            continue
        try:
            rec = run_cell(p, d)
        except Exception as exc:  # recorded, the shard moves on; a rerun retries it
            print(json.dumps({"cell": f"{p}/{d}", "status": "error", "error": repr(exc)}), flush=True)
            continue
        tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); os.replace(tmp, path)
        best = {k2: min(r["S_plant"] for r in v) for k2, v in rec["methods"].items()}
        print(json.dumps({"cell": f"{p}/{d}", "status": "ok", "seconds": round(rec["seconds"], 1),
                          "floor": rec["floor"]["S_plant"], "best": best}), flush=True)


if __name__ == "__main__":
    main()
