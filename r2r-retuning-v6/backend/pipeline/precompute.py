"""Precompute every retune the dashboard can ask for, so a click returns a stored answer.

The grid is the paper's own ranges, nothing wider:
    plants   P001 P049 P053 P158 P186 P189       (the 6 retuning plants)
    drifts   D01 .. D10                          (the 10 scenarios)
    T_log    1 2 5 10 20 50 100 ms               (Fig. 2 of the paper)
    noise    0 0.02 0.05 0.1 0.3 0.5 %           (Fig. 6 of the paper)

6 x 10 x 7 x 6 = 2,520 cells, each the adopted method (sequential-2D HGS, cost T1,
field-matched LPF 50 Hz, E_Toggle, 16 s). Results land in the SAME content-addressed
cache the live /retune route reads (`backend.pipeline.cache`, kind retune-v5), keyed by
the run spec + options -- so the dashboard needs no new code path: a click on a
precomputed combination is a cache hit, anything else computes live as before.

Resumable, single writer, paper's main point (T_log 5 ms, noise 0.3 %) first.

    python -m backend.pipeline.precompute --probe
    python -m backend.pipeline.precompute [--tlog 5] [--noise 0.003]
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline.payloads import DriftSpec, PlantSpec, ProtocolSpec, RunSpec   # noqa: E402
from backend.pipeline.retune import RetuneOptions, retune                           # noqa: E402
from backend.validation import retuning as R                                        # noqa: E402

PLANTS = ("P001", "P049", "P053", "P158", "P186", "P189")
DRIFTS = tuple(f"D{i:02d}" for i in range(1, 11))
TLOG_MS = (5.0, 1.0, 2.0, 10.0, 20.0, 50.0, 100.0)            # paper's main point first
NOISE = (0.003, 0.0, 0.0002, 0.0005, 0.001, 0.005)             # paper's main point first
STRUCTURE = "sequential-2D"
LOG = ROOT / "logs" / "precompute.jsonl"
LOCK = ROOT / "logs" / "precompute.lock"


def spec_for(pool: str, drift: str, tlog_ms: float, noise: float) -> RunSpec:
    """The RunSpec the dashboard builds for this combination; must match the UI exactly."""
    d = R.DRIFT_BY_CODE[drift]                     # scale factors: 0.5 == -50 %
    pct = lambda scale: round(100.0 * (scale - 1.0), 6)
    preset = dict(R.RETUNING_PLANTS)[pool]         # the id the UI sends (P001 -> "P01")
    return RunSpec(
        PlantSpec(source="preset", preset_id=preset),
        DriftSpec(EA_pct=pct(d.EA_scale), J_UW_pct=pct(d.J_UW_scale), J_Nip_pct=pct(d.J_Nip_scale),
                  J_RW_pct=pct(d.J_RW_scale), f_pct=pct(d.friction_scale)),
        ProtocolSpec(T_log_ms=tlog_ms, excitation="E_Toggle", record_s=16.0,
                     pct_T=noise, pct_v=noise, LPF_T_hz=50.0, LPF_v_hz=50.0,
                     Kp_star=100.0, seed=0),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tlog", type=float, default=None, help="only this T_log (ms)")
    ap.add_argument("--noise", type=float, default=None, help="only this noise fraction")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    tlogs = [a.tlog] if a.tlog is not None else list(TLOG_MS)
    noises = [a.noise] if a.noise is not None else list(NOISE)
    jobs = [(t, n, p, d) for t in tlogs for n in noises for p in PLANTS for d in DRIFTS]
    opts = RetuneOptions(gain_structure=STRUCTURE)

    if a.probe:
        t0 = time.time(); r = retune(spec_for("P001", "D07", 5.0, 0.003), opts, use_cache=True)
        print(f"probe P001/D07 5 ms 0.3 %: {time.time()-t0:.1f} s, cached={r.get('cached')}, "
              f"S={r['gains']['delivered']['on_plant']['S']:.4f}")
        print(f"grid: {len(jobs)} cells; at ~33 s/cell uncached ~ {len(jobs)*33/3600:.1f} h")
        return

    LOG.parent.mkdir(exist_ok=True)
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{LOCK} exists -- another precompute may be live")
    os.write(fd, f"pid={os.getpid()}\n".encode()); os.close(fd)
    import atexit; atexit.register(lambda: LOCK.unlink(missing_ok=True))

    t0 = time.time(); hits = 0
    print(f"{len(jobs)} cells, {STRUCTURE}, writing to the retune cache", flush=True)
    with LOG.open("a") as log:
        for i, (t, n, p, d) in enumerate(jobs, 1):
            ts = time.time()
            r = retune(spec_for(p, d, t, n), opts, use_cache=True)
            hits += bool(r.get("cached"))
            S = r["gains"]["delivered"]["on_plant"]["S"] if r.get("gains") else None
            log.write(json.dumps({"i": i, "plant": p, "drift": d, "tlog_ms": t, "noise": n,
                                  "status": r["status"], "S": S, "cached": r.get("cached"),
                                  "seconds": time.time() - ts}) + "\n"); log.flush()
            if i % 10 == 0 or r.get("cached") is False:
                print(f"  [{i}/{len(jobs)}] {t:g} ms {100*n:.2f} % {p}/{d}  {r['status']}  "
                      f"S={S if S is None else round(S,4)}  {'cache' if r.get('cached') else f'{time.time()-ts:.0f}s'}",
                      flush=True)
    print(f"\ndone: {len(jobs)} cells, {hits} already cached, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
