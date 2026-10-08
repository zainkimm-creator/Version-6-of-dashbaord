# Plan — T1 cost, two-stage per-zone gains, out-feeder BO campaign (7 Oct 2026)

**Spec (user, session 12):**
1. Dashboard scores retunes with the repaired cost (T1), not the paper's Eq. (12) (T0).
2. Only two gain structures: (a) a free (K_p*, T_I) per zone (6 gains); (b) UW and RW fixed,
   out-feeder searched. The out-feeder result is shown on the dashboard; the 6 per-zone gains
   are saved in the backend and reused by the next case.
3. The dashboard front end reflects every recent backend change.
4. Run the BO campaign fully on GPU, with a spec file in the detail of the user's example
   (search space, gp_minimize call, warm-start star, seed/random/EI split, shared random
   designs), and report it BEFORE anything goes into the front end.

No git repo: no commits; ledger at `.superpowers/sdd/2026-10-07-t1-two-stage-bo/progress.md`.
Every task: failing test first (where code), full suite green before the task is done.

## Global constraints
- Do not touch `backend/validation/*` kernels (T0 identity gates, goldens).
- `.venv`, `frontend/node_modules` are symlinks into V5 — never delete.
- GPU jobs: export the venv's nvidia lib dirs on LD_LIBRARY_PATH, XLA_PYTHON_CLIENT_PREALLOCATE=false,
  and assert `jax.devices()` is a CUDA device before running (the 2026-10-06 paper-methods run
  silently fell back to CPU).
- Precompute stopped by the user's decision; relaunch under the new defaults after the BO.

## Tasks

### Task 1 — BO campaign script + spec file (GPU)  [runs in background while Tasks 2-6 proceed]
- `multiloop/src/run_bo_outfeeder.py`: 6 plants x 10 drifts, tier T1, field-matched SysID.
  Per plant: commissioning = full-6D twin search on the PRE-DRIFT plant's twin (seed 0) -> the
  saved 6 gains. Per cell: UW, RW frozen at the saved gains; out-feeder arms
  HGS-only, HGS+BO(5), HGS+BO(10), CS-BO(30) s0-2, WS-BO(30) s0-2 (BO = `run_paper_methods._bo`,
  unchanged arithmetic); floors: out-feeder floor (HGS on the true plant, ends frozen) and the
  6-D floor (generic search on the true plant); also the commissioned 6 gains on the drifted plant
  (no retune). Resumable per-cell JSON, lock, N GPU worker processes, GPU assertion.
- `multiloop/reports/BO-OUTFEEDER-SPEC.md`: the exact procedure, with numbers verified by code
  (skopt 0.10.2 split, shared random designs per seed).
- Gate: probe cell P001/D01 per-cell-frozen variant on GPU vs the stored CPU result.
- Expected: 60/60 cells ok.

### Task 2 — T1 is the dashboard's cost
- `retune.DEFAULT_COST_TIER = "T1"`; `/plant/baseline` and `/retune` defaults follow it.
- Tests: default is T1; a retune and baseline without a tier report `cost_tier == "T1"`.
- Golden/default-pinned tests that move: list each, why, new value.

### Task 3 — zone-gains store
- `backend/pipeline/zone_gains.py`: `key(plant_spec, tier)`, `load`, `save`, `delete`; JSON under
  `data/session/zone_gains/`. Tests: round trip, per-tier separation, delete.

### Task 4 — two-stage retune
- `gain_structure="full-6D"`: 6-D twin search; saves the 6 gains to the store (fresh and cache hit).
- `gain_structure="outfeeder-only-2D"` (new default): UW/RW frozen at the stored gains (if none:
  run full-6D for this case first and store it); authors' 2-D HGS on the out-feeder.
  Payload: `search.per_zone_gains` {kp, ti, S_twin, source run, reused}. Cache key includes the
  frozen ends.
- `DEFAULT_GAIN_STRUCTURE = "outfeeder-only-2D"`. Step-7 restore unchanged.
- Tests: first call stores; second case reuses (no 6-D search, UW/RW equal stored); full-6D
  overwrites; cache key differs when the stored gains differ.

### Task 5 — API
- `GET /zone-gains` (run_spec, cost_tier) -> stored or null; `DELETE /zone-gains`. Tests.

### Task 6 — front end
- Selector: the two structures only. Twin window: the out-feeder result + the stored 6 gains
  (source case, reused or new), "re-commission (6 gains)" = run full-6D. Cost labels say T1.
- Build clean; screenshot inspected.

### Task 7 — BO report (before any front-end use)
- `multiloop/src/aggregate_bo_outfeeder.py` -> tables + one figure; `BO-OUTFEEDER-RESULTS.md/pdf`.

### Task 8 — relaunch precompute under the new defaults; final review; PROGRESS.md.
