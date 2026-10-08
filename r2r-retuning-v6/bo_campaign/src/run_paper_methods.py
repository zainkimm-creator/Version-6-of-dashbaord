"""The paper's five retuning methods, under the repaired cost, for two gain structures.

Implements `reports/PAPER-METHODS-PLAN.md` (from `Changes_to_be_done_in_the_dashboard.docx`).

Structures (run separately, compared afterwards):
    outfeeder-only   UW and RW FIXED at their own values from the PER-ZONE twin
                     optimum (the 6-D search is run first, its UW and RW entries are
                     frozen), then only the out-feeder is optimised. 2-D, so HGS-only
                     uses the paper's actual hierarchical grid search. (User's
                     instruction of 2026-10-05: "run the six gains for each motor
                     first, then choose those optima for UW and RW, then find the
                     out-feeder one".)
    perzone          a free (K_p*, T_I) per tension zone. 6-D, so HGS-only uses the
                     dimension-generic twin search (a grid does not exist in 6-D).

Methods per (cell, structure), scored on the TRUE drifted plant under tier T1:
    HGS-only      0 real evals   twin search only                      seed 0
    HGS+BO(5)     5              BO on the plant, warm from HGS answer  seed 0
    HGS+BO(10)   10              same                                   seed 0
    CS-BO(30)    30              BO on the plant, cold start            seeds 0,1,2
    WS-BO(30)    30              BO on the plant, warm from the SysID   seeds 0,1,2
                                 operating point (100, T_I(theta-hat)),
                                 NOT the twin's optimum
    floor         -              the structure searched on the real plant

"Real evals" exclude the one-time SysID acquisition run, as the document specifies.

    python src/run_paper_methods.py --probe
    python src/run_paper_methods.py
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
for q in (str(REPO), str(HERE / "src")):
    if q not in sys.path:
        sys.path.insert(0, q)

from backend.validation import retuning as R                            # noqa: E402
from backend.validation import retuning_paper_protocol as AP            # noqa: E402
from backend.validation.plants import parameters_for_plant              # noqa: E402
from backend.validation.retuning_structures import BY_NAME, Structure   # noqa: E402
from backend.validation.retuning_structure_search import search         # noqa: E402
from backend.validation.retuning_tiered_eval import ROLLER_NAMES, TieredEvaluator  # noqa: E402

OUT = HERE / "reports" / "paper_cells"
TIER = "T1"
PLANTS = ("P001", "P049", "P053", "P158", "P186", "P189")
DRIFTS = tuple(f"D{i:02d}" for i in range(1, 11))
STRUCTURES = {"outfeeder-only": BY_NAME["outfeeder-only-2D"], "perzone": BY_NAME["full-6D"]}
SYSID_KP = 100.0                    # the SysID-mode operating point's K_p*
HGS_BO_BUDGETS = (5, 10)
BO30 = 30
BO_SEEDS = (0, 1, 2)
STAR_STEP = 0.30


# ----------------------------------------------------------------------------- #
# BO over a structure's own search vector (2-D or 6-D), on the REAL plant
# ----------------------------------------------------------------------------- #
def _star_x(x: np.ndarray, st: Structure, limit=None):
    """Incumbent, then each coordinate +/-30 % (clipped) -- V5's warm-start set."""
    lo = np.array([b[0] for b in st.bounds]); hi = np.array([b[1] for b in st.bounds])
    pts = [np.clip(x.copy(), lo, hi)]
    for j in range(st.dim):
        for f in (1 - STAR_STEP, 1 + STAR_STEP):
            y = x.copy(); y[j] = float(np.clip(y[j] * f, lo[j], hi[j])); pts.append(y)
    return pts[:limit] if limit else pts


def _bo(plant_costs, st: Structure, baseline, budget, warm_x=None, seed=0):
    """Returns (kp3, ti3, best_cost, real_evals)."""
    from skopt import gp_minimize
    from skopt.space import Real

    space = [Real(b[0], b[1], prior="log-uniform" if b[2] else "uniform", name=f"x{j}")
             for j, b in enumerate(st.bounds)]

    def cost_of_x(x):
        kp, ti = st.expand_batch(np.asarray(x, dtype=float)[None, :], baseline)
        c = float(plant_costs(kp, ti)[0])
        return c if np.isfinite(c) else 1e3

    x0 = y0 = None
    if warm_x is not None:
        keep = max(1, budget // 2)
        pts = _star_x(np.asarray(warm_x, dtype=float), st, limit=keep)
        x0 = [[float(v) for v in p] for p in pts]
        y0 = [cost_of_x(p) for p in pts]
    remaining = max(1, budget - (len(x0) if x0 else 0))
    n_init = max(3, min(10, remaining // 3))
    res = gp_minimize(cost_of_x, space, n_calls=remaining,
                      n_initial_points=min(n_init, remaining), x0=x0, y0=y0,
                      acq_func="EI", noise=1e-6, random_state=seed)
    kp, ti = st.expand_batch(np.asarray(res.x, dtype=float)[None, :], baseline)
    return kp[0], ti[0], float(res.fun), budget


# ----------------------------------------------------------------------------- #
# the twin search: the paper's HGS where a grid exists (2-D), generic otherwise
# ----------------------------------------------------------------------------- #
def _twin_search(costs, st: Structure, baseline, seed):
    """-> (x, kp3, ti3, cost, evaluations, how)."""
    if st.dim == 2:
        def batch(kps, tis):
            x = np.stack([np.asarray(kps, float), np.asarray(tis, float)], axis=1)
            kp, ti = st.expand_batch(x, baseline)
            return costs(kp, ti)
        h = AP.hierarchical_grid_search(batch)
        x = np.array([h.kp, h.ti], dtype=float)
        kp, ti = st.expand_batch(x[None, :], baseline)
        return x, kp[0], ti[0], float(h.cost), int(h.evaluations), "HGS (authors')"
    r = search(costs, st, baseline=baseline, seed=seed)
    return r.x, r.kp, r.ti, float(r.cost), int(r.evaluations), "generic twin search (6-D)"


def run_cell(pool, drift, sname):
    t0 = time.time()
    st = STRUCTURES[sname]
    base, meta = parameters_for_plant(dict(R.RETUNING_PLANTS)[pool])
    v0 = float(meta.get("v0_mps") or base.feeder_velocity_m_s)
    plant = R.apply_drift(base, R.DRIFT_BY_CODE[drift])
    twin, fit = R.identify_twin(plant, R.PROTOCOLS["field_matched"], meta)

    # the SysID operating point: K_p* = 100 on every zone, T_I from theta-hat
    ti_hat = float(R.plant_auto_ti_s(twin, v0))
    op_kp, op_ti = np.full(3, SYSID_KP), np.full(3, ti_hat)

    tw = TieredEvaluator(twin, v0, tier=TIER)
    pl = TieredEvaluator(plant, v0, tier=TIER)

    # Where the frozen zones sit for outfeeder-only: at THEIR OWN values from the
    # per-zone twin optimum. Same deterministic 6-D search, same seed, as the
    # perzone cell's HGS-only, so the two structures share one starting point.
    baseline = None
    frozen_from = None
    if st.needs_baseline:
        _, zkp, zti, zcost, zn, _ = _twin_search(tw.costs, STRUCTURES["perzone"], None, seed=0)
        baseline = (zkp.copy(), zti.copy())
        frozen_from = {"kp": zkp.tolist(), "ti": zti.tolist(), "S_twin": zcost,
                       "twin_evals": zn, "note": "UW and RW frozen at these; nip re-optimised"}
    # The same point expressed in the structure's own search vector -- CLIPPED to
    # the paper's box. T_I(theta-hat) is 36-50 s on P049/P053/P158/P186, above the
    # 30 s ceiling of the box the document also specifies, and skopt refuses a warm
    # start outside the space (that is how the first launch died at unit 11). The
    # raw value and the clipped value are both recorded in the cell.
    lo = np.array([b[0] for b in st.bounds]); hi = np.array([b[1] for b in st.bounds])
    op_x_raw = (np.array([SYSID_KP, ti_hat]) if st.dim == 2
                else np.array([SYSID_KP] * 3 + [ti_hat] * 3))
    op_x = np.clip(op_x_raw, lo, hi)
    ti_hat_used = float(np.clip(ti_hat, 0.5, 30.0))

    on_plant = lambda kp, ti: float(pl.costs(np.asarray(kp)[None, :], np.asarray(ti)[None, :])[0])

    methods = {}
    # HGS-only
    x, kp, ti, s_twin, n_tw, how = _twin_search(tw.costs, st, baseline, seed=0)
    s_hgs = on_plant(kp, ti)
    methods["HGS-only"] = [{"seed": 0, "kp": kp.tolist(), "ti": ti.tolist(),
                            "S_twin": s_twin, "S_plant": s_hgs, "real_evals": 0,
                            "twin_evals": n_tw, "search": how}]
    # HGS+BO(5), HGS+BO(10): warm from the HGS answer, seed 0 only (deterministic family)
    for b in HGS_BO_BUDGETS:
        bkp, bti, c, n = _bo(pl.costs, st, baseline, b, warm_x=x, seed=0)
        methods[f"HGS+BO({b})"] = [{"seed": 0, "kp": bkp.tolist(), "ti": bti.tolist(),
                                    "S_plant": min(c, s_hgs), "S_plant_raw": c,
                                    "real_evals": n, "improved_on_hgs": bool(c < s_hgs)}]
    # CS-BO(30) and WS-BO(30): 3 seeds each
    methods["CS-BO(30)"], methods["WS-BO(30)"] = [], []
    for seed in BO_SEEDS:
        ckp, cti, c, n = _bo(pl.costs, st, baseline, BO30, warm_x=None, seed=seed)
        methods["CS-BO(30)"].append({"seed": seed, "kp": ckp.tolist(), "ti": cti.tolist(),
                                     "S_plant": c, "real_evals": n})
        wkp, wti, c, n = _bo(pl.costs, st, baseline, BO30, warm_x=op_x, seed=seed)
        methods["WS-BO(30)"].append({"seed": seed, "kp": wkp.tolist(), "ti": wti.tolist(),
                                     "S_plant": c, "real_evals": n,
                                     "warm_start": {"kp": op_kp.tolist(), "ti": op_ti.tolist(),
                                                    "ti_used_s": ti_hat_used,
                                                    "clipped_to_box": bool(ti_hat_used != ti_hat)}})
    # floor
    fx, fkp, fti, fcost, fn, fhow = _twin_search(pl.costs, st, baseline, seed=7)

    return {"pool": pool, "drift": drift, "structure": sname, "tier": TIER,
            "roller_names": list(ROLLER_NAMES),
            "twin_mare_percent": float(fit["mare_theta_percent"]),
            "sysid_operating_point": {"kp": op_kp.tolist(), "ti": op_ti.tolist(),
                                      "ti_from_theta_hat_s": ti_hat},
            "sysid_operating_point_S_plant": on_plant(op_kp, op_ti),
            "frozen_zones_from_perzone_optimum": frozen_from,
            "methods": methods,
            "floor": {"kp": fkp.tolist(), "ti": fti.tolist(), "S_plant": fcost, "search": fhow},
            "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", default=",".join(PLANTS))
    ap.add_argument("--drifts", default=",".join(DRIFTS))
    ap.add_argument("--structures", default=",".join(STRUCTURES))
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    pools = [p for p in args.plants.split(",") if p]
    drifts = [d for d in args.drifts.split(",") if d]
    structs = [s for s in args.structures.split(",") if s]

    if args.probe:
        for sname in structs:
            t = time.time()
            rec = run_cell(pools[0], drifts[0], sname)
            dt = time.time() - t
            print(f"probe {pools[0]}/{drifts[0]} {sname}: {dt:.1f} s")
            print(f"    SysID op point: K_p*=100, T_I(theta-hat)={rec['sysid_operating_point']['ti_from_theta_hat_s']:.3f} s"
                  f"  -> S_plant={rec['sysid_operating_point_S_plant']:.5f}")
            for m, runs in rec["methods"].items():
                ss = ", ".join(f"{r['S_plant']:.5f}" for r in runs)
                extra = f"  [{runs[0]['search']}, {runs[0]['twin_evals']:,} twin evals]" if m == "HGS-only" else ""
                print(f"    {m:<12} real={runs[0]['real_evals']:<3} S_plant={ss}{extra}")
            print(f"    {'floor':<12} real=--  S_plant={rec['floor']['S_plant']:.5f}")
            print(f"  -> 60 cells x this structure ~ {dt * 60 / 3600:.1f} h")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT.parent / "paper_methods.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{lock} exists -- another run may be live.")
    os.write(fd, f"pid={os.getpid()}\n".encode()); os.close(fd)
    import atexit
    atexit.register(lambda: lock.unlink(missing_ok=True))

    jobs = [(s, p, d) for s in structs for p in pools for d in drifts]
    print(f"{len(jobs)} units = {len(structs)} structures x {len(pools) * len(drifts)} cells", flush=True)
    t0 = time.time()
    for i, (sname, pool, drift) in enumerate(jobs, 1):
        f = OUT / f"{pool}__{drift}__{sname}__s0.json"
        if f.exists():
            print(f"  [{i}/{len(jobs)}] {sname} {pool}/{drift} cached", flush=True); continue
        rec = run_cell(pool, drift, sname)
        f.write_text(json.dumps(rec, indent=1))
        m = rec["methods"]
        med = lambda k: float(np.median([r["S_plant"] for r in m[k]]))
        print(f"  [{i}/{len(jobs)}] {sname} {pool}/{drift}  HGS {med('HGS-only'):.4f}  "
              f"+BO10 {med('HGS+BO(10)'):.4f}  CS {med('CS-BO(30)'):.4f}  WS {med('WS-BO(30)'):.4f}"
              f"  floor {rec['floor']['S_plant']:.4f}  [{rec['seconds']:.0f}s]", flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
