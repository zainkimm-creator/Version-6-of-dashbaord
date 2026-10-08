"""Summarise the out-feeder BO campaign (reports/bo_outfeeder_cells/*.json).

    JAX_PLATFORMS=cpu ../.venv/bin/python src/aggregate_bo_outfeeder.py

Writes tables/bo_outfeeder_summary.csv, tables/bo_outfeeder_per_plant.csv,
tables/bo_outfeeder_runs.csv and figures/bo_outfeeder_gap.png.

Statistics: every run is one (cell, seed); the median is over runs (60 for the single-seed
arms, 180 for CS-BO / WS-BO). Gap to floor = S_arm / S_floor - 1, where the floor is the
authors' HGS on the TRUE drifted plant with the same frozen ends.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
CELLS = HERE / "reports" / "bo_outfeeder_cells"
TABLES = HERE / "tables"
FIGURES = HERE / "figures"
ARMS = ("commissioned-6", "HGS-only", "HGS+BO(5)", "HGS+BO(10)", "WS-BO(30)", "CS-BO(30)")


def _pct(a, b):
    return 100.0 * (a / b - 1.0)


def _runs(cells):
    for c in cells:
        floor = c["floor"]["S_plant"]
        comm = c["methods"]["commissioned-6"][0]["S_plant"]
        hgs = c["methods"]["HGS-only"][0]["S_plant"]
        for arm in ARMS:
            for r in c["methods"][arm]:
                s = r["S_plant"]
                yield {"pool": c["pool"], "drift": c["drift"], "arm": arm, "seed": r["seed"],
                       "S_plant": s, "real_evals": r["real_evals"],
                       "gap_to_floor_pct": _pct(s, floor), "change_vs_commissioned_pct": _pct(s, comm),
                       "beats_hgs_only": bool(s < hgs)}


def _arm_stats(rows):
    g = np.array([r["gap_to_floor_pct"] for r in rows])
    ch = np.array([r["change_vs_commissioned_pct"] for r in rows])
    return {"runs": len(rows), "real_evals": rows[0]["real_evals"],
            "median_gap_to_floor_pct": float(np.median(g)),
            "q25_gap_to_floor_pct": float(np.percentile(g, 25)),
            "q75_gap_to_floor_pct": float(np.percentile(g, 75)),
            "max_gap_to_floor_pct": float(np.max(g)),
            "median_change_vs_commissioned_pct": float(np.median(ch)),
            "worse_than_commissioned": int(np.sum(ch > 1e-9)),
            "beats_hgs_only": int(sum(r["beats_hgs_only"] for r in rows)),
            "within_1pct_of_floor": int(np.sum(g <= 1.0)),
            "below_floor": int(np.sum(g < -1e-9))}


def summarise(cells):
    rows = list(_runs(cells))
    arms = {a: _arm_stats([r for r in rows if r["arm"] == a]) for a in ARMS}
    price = [_pct(c["floor"]["S_plant"], c["floor_6d"]["S_plant"]) for c in cells if c.get("floor_6d")]
    plants = sorted({c["pool"] for c in cells})
    per_plant = {p: {a: _arm_stats([r for r in rows if r["arm"] == a and r["pool"] == p]) for a in ARMS}
                 for p in plants}
    return {"cells": len(cells), "arms": arms, "per_plant": per_plant, "rows": rows,
            "frozen_ends_price_pct": float(np.median(price)) if price else None,
            "frozen_ends_price_per_cell": price}


def load():
    return [json.loads(p.read_text()) for p in sorted(CELLS.glob("P*__D*.json"))]


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def figure(s, path):
    """Box plot of the gap to the HGS floor per arm, symmetric-log axis: runs BELOW the
    floor (negative gaps) are drawn where they are, not clipped."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.8,
                         "xtick.direction": "in", "ytick.direction": "in"})
    colours = {"commissioned-6": "#8c8c8c", "HGS-only": "#0072B2", "HGS+BO(5)": "#56B4E9",
               "HGS+BO(10)": "#009E73", "WS-BO(30)": "#E69F00", "CS-BO(30)": "#D55E00"}
    hatches = {"commissioned-6": "", "HGS-only": "//", "HGS+BO(5)": "..", "HGS+BO(10)": "xx",
               "WS-BO(30)": "\\\\", "CS-BO(30)": "oo"}
    data = [[r["gap_to_floor_pct"] for r in s["rows"] if r["arm"] == a] for a in ARMS]
    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=200)
    bp = ax.boxplot(data, positions=range(len(ARMS)), widths=0.55, patch_artist=True,
                    whis=(0, 100), medianprops={"color": "black", "linewidth": 1.4})
    for patch, a in zip(bp["boxes"], ARMS):
        patch.set_facecolor(colours[a]); patch.set_hatch(hatches[a]); patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_yscale("symlog", linthresh=0.1, linscale=0.6)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([f"{a}\n{s['arms'][a]['real_evals']} line runs\nmedian {s['arms'][a]['median_gap_to_floor_pct']:.2f} %"
                        for a in ARMS], fontsize=7.5)
    ax.set_ylabel("cost above the HGS floor (%)\nbox 25–75 %, whiskers min–max")
    lo = min(min(d) for d in data); hi = max(max(d) for d in data)
    ax.set_ylim(min(-0.05, lo * 2.0), hi * 2.5)
    ax.tick_params(top=True, right=True, which="both")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    cells = load()
    if not cells:
        sys.exit("no cells yet")
    s = summarise(cells)
    _write_csv(TABLES / "bo_outfeeder_runs.csv", s["rows"])
    _write_csv(TABLES / "bo_outfeeder_summary.csv",
               [{"arm": a, **v} for a, v in s["arms"].items()])
    _write_csv(TABLES / "bo_outfeeder_per_plant.csv",
               [{"plant": p, "arm": a, **v} for p, d in s["per_plant"].items() for a, v in d.items()])
    figure(s, FIGURES / "bo_outfeeder_gap.png")
    print(json.dumps({"cells": s["cells"], "frozen_ends_price_pct": s["frozen_ends_price_pct"],
                      "arms": s["arms"]}, indent=1))


if __name__ == "__main__":
    main()
