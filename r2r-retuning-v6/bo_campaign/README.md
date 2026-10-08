# bo_campaign — out-feeder BO campaign (7–8 Oct 2026)

**The question this answers:** once UW and RW are frozen at a plant's commissioned six gains, which way of retuning the **out-feeder's (K_p\*, T_I)** gets closest to the best achievable cost? The campaign compares the paper's twin search (HGS), few-shot BO on the line warm-started from HGS, and 30-run BO without the twin. Everything is simulated: a "line run" is one evaluation on the simulated true drifted plant.

- **Cells:** 6 plants (P001, P049, P053, P158, P186, P189 = dashboard P01, P02, P03, P06, P09, P10) × 10 drifts (D01–D10) = **60 cells**. Each cell identifies its own twin of the drifted plant (field-matched protocol).
- **Cost:** tier **T1**, the repaired cost (`backend/validation/retuning_tiered_eval.py`). It is scored on the **true drifted plant**. A diverging run is fed to the optimiser as 1000.
- **Frozen ends:** UW and RW come from `reports/bo_outfeeder_cells/commissioning/<plant>.json`, which is the 6-D twin search on the **pre-drift** twin (seed 0, 12,175 twin evaluations).
- **Hardware:** GPU (RTX 5070 Ti). Every cell records `"device": "cuda:0"`.
- **Full specification:** [`reports/BO-OUTFEEDER-SPEC.md`](reports/BO-OUTFEEDER-SPEC.md). **Results:** [`reports/BO-OUTFEEDER-RESULTS.md`](reports/BO-OUTFEEDER-RESULTS.md) (PDFs alongside).

## Layout

The layout mirrors the original `multiloop/` folder, so the scripts run unchanged. `HERE` is `bo_campaign/` and `REPO` is the bundle root; the backend is imported from `REPO`.

| path | what |
|---|---|
| `src/run_bo_outfeeder.py` | the campaign: `--commission`, then one worker per `--shard i/N`, or `--only PLANT/DRIFT` |
| `src/run_paper_methods.py` | the 6 Oct paper-methods campaign; supplies `_twin_search` (HGS in 2-D, generic search in 6-D) and the `--probe` used for the GPU/CPU gate |
| `src/launch_bo_outfeeder.sh` | GPU launcher: commission 6 plants in sequence, then N shard workers |
| `src/aggregate_bo_outfeeder.py` | 60 cells → `tables/*.csv` + `figures/bo_outfeeder_gap.png` |
| `src/tiers.py`, `src/paper_t0.py` | the study's copy of the cost tiers, pinned equal to the backend by `tests/test_tiered_eval_matches_study.py` (0 difference) |
| `tests/test_bo_outfeeder_spec.py` | the split, star, shared random designs and GPU guard (11 cases) |
| `tests/test_aggregate_bo_outfeeder.py` | the summary arithmetic (6 cases) |
| `reports/bo_outfeeder_cells/P*__D*.json` | **60 result cells**: every arm, every seed, the BO trajectories, the floor, the 6-D floor, device, seconds |
| `reports/bo_outfeeder_cells/commissioning/*.json` | the saved six gains per plant (the frozen UW/RW) |
| `reports/bo_outfeeder_cells/gpu_cpu_gate_P001_D01.log` | the GPU-vs-CPU parity record of the reproducibility gate (below); kept on purpose, it holds no paths |
| `reports/paper_cells/` | 120 cells of the 6 Oct campaign (`outfeeder-only` and `perzone`, CPU). They are the source of the **6-D floor** and of the gate's CPU reference |
| `tables/`, `figures/` | aggregated outputs |

## The arms, and how they are built on HGS and skopt

| arm | line runs | seeds | how it is built |
|---|---|---|---|
| `commissioned-6` | 0 | — | the saved six gains, unchanged ("doing nothing") |
| `HGS-only` | 0 | 0 | `_twin_search` (`src/run_paper_methods.py:108`) runs the authors' `hierarchical_grid_search` (`backend/validation/retuning_paper_protocol.py:137`) on the **twin**, with UW and RW frozen: 2,805 twin evaluations (30×30 coarse, 1,000 LHS, 30×30 fine, 5 polish) over K_p\* ∈ [1, 500] (log) and T_I ∈ [0.5, 30] s (`run_bo_outfeeder.py:183`). This is plain HGS, without the dashboard's keep-incumbent guard. |
| `HGS+BO(5)`, `HGS+BO(10)` | 5, 10 | 0 | skopt `gp_minimize` on the **true plant**, warm-started with the ±30 % star around the HGS answer. Each reports min(BO, HGS-only) (`:187–192`). |
| `CS-BO(30)` | 30 | 0, 1, 2 | `gp_minimize`, cold start (`:195`) |
| `WS-BO(30)` | 30 | 0, 1, 2 | `gp_minimize`, warm star around the SysID operating point (K_p\* = 100, T_I = T_I(θ̂) clipped to 30 s) (`:198`) |
| **floor** | — | — | the same HGS on the **true** drifted plant, with the same frozen ends (`:204`). Every gap is measured from it. It is a grid search, so a BO run can land slightly below it. |
| **6-D floor** | — | — | a free six-gain search on the true plant, read from `reports/paper_cells/<plant>__<drift>__perzone__s0.json` (`:205–206`). It prices the freezing. |

**The `gp_minimize` call** (`src/run_bo_outfeeder.py:106`, identical for every BO arm):

```python
gp_minimize(cost_of_x, space, n_calls=remaining, n_initial_points=n_rnd,
            x0=x0, y0=y0, acq_func="EI", noise=1e-6, random_state=seed)
# space: Real(1, 500, prior="log-uniform") for K_p*,  Real(0.5, 30, prior="uniform") for T_I [s]
# x0 = star(centre)[:min(budget // 2, 5)]   (centre, K_p* -30 %/+30 %, T_I -30 %/+30 %, clipped)
# y0 = their costs;  remaining = max(3, budget - len(x0));  n_rnd = max(3, min(10, remaining // 3))
```

The surrogate is skopt's default: Matérn 5/2 with ARD × a constant kernel, `normalize_y=True`, and 2 optimiser restarts (verified in skopt's `cook_estimator`, see the spec §2).

**Seed / random / EI split.** skopt 0.10.2 adds `n_initial_points` on top of `len(x0)` (see the spec §2), so the split is as below. This table is `run_bo_outfeeder.bo_split` and applies to this campaign only. The 6 Oct `paper_cells/` were made by `run_paper_methods._bo` (`:77`, warm start at `:92`), which keeps `max(1, budget // 2)` star points (about 15 for WS-BO(30)); from them this campaign uses only the 6-D floor and the gate.

| arm | budget | star seeds | random | EI | warm start from |
|---|---|---|---|---|---|
| CS-BO(30) | 30 | 0 | 10 | 20 | — (cold) |
| WS-BO(30) | 30 | 5 | 8 | 17 | SysID point: K_p\* = 100, T_I = T_I(θ̂) clipped to 30 s |
| HGS+BO(5) | 5 | 2 | 3 | 0 | the twin's HGS answer |
| HGS+BO(10) | 10 | 5 | 3 | 2 | the twin's HGS answer |

`random_state` depends only on the seed, so the campaign has exactly **three random designs**, shared by every cell and arm. For seed 0 the design starts (39.817, 25.406), (206.809, 25.494), (48.193, 11.839) (`SEED0_RANDOM_DESIGN`, `:56`). The results report states that all 60 cells match these designs, with 0 mismatches.

## Re-running

The runner needs the GPU: `require_gpu` (`:126`) refuses to start on CPU, because an earlier run had silently fallen back to CPU. Install `requirements-gpu.txt` into `.venv/` at the bundle root, then run from `bo_campaign/`:

```bash
export LD_LIBRARY_PATH="$(ls -d "$PWD"/../.venv/lib/python3*/site-packages/nvidia/*/lib | tr '\n' ':')" XLA_PYTHON_CLIENT_PREALLOCATE=false
```

**All 60 cells.** Results are cached per file, so first move the shipped cells and the commissioning files aside:

```bash
mkdir -p /tmp/bo_shipped && mv reports/bo_outfeeder_cells/P*__D*.json reports/bo_outfeeder_cells/commissioning /tmp/bo_shipped/
bash src/launch_bo_outfeeder.sh 6        # commission 6 plants (in sequence), then 6 shard workers
```

- Commissioning runs one plant at a time, because six parallel 6-D searches ran a 16 GB GPU out of memory.
- Worker stderr goes to `reports/bo_outfeeder_cells/stderr/`; progress lines (one JSON per cell) go to stdout.
- With six workers sharing one GPU, cells took 852–1,275 s each (the `seconds` field).
- A cell that errors is logged and skipped; rerunning the launcher retries only the missing files.
- Keep the shipped commissioning files instead of moving them if you want to reuse the original frozen ends.

**One cell:**

```bash
mv reports/bo_outfeeder_cells/P001__D01.json /tmp/            # otherwise it is reported "cached"
../.venv/bin/python src/run_bo_outfeeder.py --only P001/D01   # needs commissioning/P001.json
```

Then compare `methods.<arm>[*].S_plant` and `floor.S_plant` with the moved file. The comparison snippet is in `REVIEW_GUIDE.md` §5.

## Aggregating

```bash
JAX_PLATFORMS=cpu ../.venv/bin/python src/aggregate_bo_outfeeder.py
```

This reads `reports/bo_outfeeder_cells/P*__D*.json` and **overwrites** four outputs:
- `tables/bo_outfeeder_summary.csv` (per arm);
- `tables/bo_outfeeder_per_plant.csv`;
- `tables/bo_outfeeder_runs.csv` (one row per cell × arm × seed);
- `figures/bo_outfeeder_gap.png`.

It also prints the summary as JSON. To check the shipped tables, copy `tables/` first and `diff` afterwards.

The statistics work as follows:
- Gap is S / S_floor − 1.
- Medians are over runs: 60 for single-seed arms, 180 for CS/WS-BO.
- Runs below the floor are counted, not clipped.
- The "price of freezing" is floor / 6-D floor − 1.

## Reproducibility gate

Before the full launch, cell **P001/D01** of the 6 Oct per-cell-frozen variant (`run_paper_methods.py`, structure `outfeeder-only`) was re-run on the GPU. It was compared with the stored CPU result in `reports/paper_cells/P001__D01__outfeeder-only__s0.json`. Every arm and the floor matched to 6 decimals (HGS-only 0.519386; floor 0.5192223732921541 on both; 227 s). The record is `reports/bo_outfeeder_cells/gpu_cpu_gate_P001_D01.log`. Its GPU-vs-CPU side-by-side format was written by a one-off comparison script that is not shipped; `--probe` (below) prints one device's values only.

The gate's numbers belong to that variant, whose ends are frozen at the drifted twin's own 6-D optimum. They are **not** the final campaign's P001/D01 values. To repeat the gate (it runs on CPU or GPU and writes nothing):

```bash
../.venv/bin/python src/run_paper_methods.py --probe --plants P001 --drifts D01 --structures outfeeder-only
```

It prints each arm's S_plant to 5 decimals for the device it runs on. Compare them by hand with `reports/paper_cells/P001__D01__outfeeder-only__s0.json`; agreement to the 5 printed decimals is what you can check this way.

## Outputs and headline

Headline from `tables/bo_outfeeder_summary.csv` (gap to floor, %):

| arm | line runs | median | worst |
|---|---|---|---|
| HGS-only | 0 | 0.062 | 22.531 |
| HGS+BO(5) | 5 | 0.060 | 3.435 |
| HGS+BO(10) | 10 | 0.056 | 3.244 |
| CS-BO(30) | 30 | 0.081 | 7.519 |
| WS-BO(30) | 30 | 0.169 | 7.574 |
| doing nothing | 0 | 1.632 | 83.145 |

Freezing UW and RW at commissioning costs a median +2.11 % against the 6-D floor (P189's own median is 21.8 %; the per-cell maximum is +423 % on P186 D07). The interpretation, worst cases and recommendation are in `reports/BO-OUTFEEDER-RESULTS.md`.
