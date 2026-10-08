# r2r-retuning-v6 — review bundle (8 Oct 2026)

**What this is.**
- A reproduction of the paper's roll-to-roll (R2R) tension retuning (Algorithm 1: identify the drifted line, build a digital twin, retune on the twin, validate and restore). It ships as a FastAPI backend plus a React/Vite dashboard.
- The bundle holds the backend exactly as it runs today: the repaired cost **T1** is the default, and the paper's **Eq. (12) (T0)** is still selectable and bit-identical. It also holds the **two-stage per-zone gains**, the **step-7 restore (A9)**, the authors' **hierarchical grid search (HGS, 2,805 evaluations)** and the **6-D structure search**.
- `bo_campaign/` holds the **out-feeder Bayesian-optimisation (BO) campaign** (6 plants × 10 drifts = 60 cells) that compares HGS-only, HGS+BO, CS-BO and WS-BO under T1. It has the code, the spec, the 60 result files and the aggregated tables.
- Everything here is simulation. No gain in this bundle has been run on a physical line.
- **Start with [`REVIEW_GUIDE.md`](REVIEW_GUIDE.md)**. It covers what changed and why, where each piece lives (file:line), and how to check each claim.

## Install

Python 3.11 and Node.js 20.19+ or 22.12+ (Vite 8.0.16's requirement; produced with Node 22). Run everything from the bundle root.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt      # CPU-only JAX
# or, on an NVIDIA GPU (CUDA 12):
.venv/bin/python -m pip install -r requirements-gpu.txt -r requirements-dev.txt
(cd frontend && npm ci)
```

The versions are pinned to the environment that produced every result here: JAX 0.10.2, scikit-optimize 0.10.2, scikit-learn 1.9.0, NumPy 2.4.6, SciPy 1.17.1. The BO campaign needs the GPU build because its runner refuses to start on CPU. The backend and the tests run on CPU.

## Run the tests

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q                 # fast set (pytest.ini excludes -m slow)
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q -m slow         # slow set: validation-route goldens, end-to-end /retune
(cd bo_campaign && JAX_PLATFORMS=cpu ../.venv/bin/python -m pytest -q tests)   # BO spec + aggregation
```

The fast set includes real end-to-end retunes (a 6-D search on CPU). On a fresh unzip of this bundle it took about 10 min on CPU: **430 passed, 66 skipped, 0 failed**, the same counts as the source project. The 66 skips depend on the environment:
- 22 in `tests/test_retuning_jax.py` need `JAX_ENABLE_X64=1`;
- 44 in `tests/test_retuning_tier1.py` need the external v5 figure package, looked up as `paper1_v5_figure_package/` in the bundle root's grandparent directory (`PROJECT_ROOT.parents[1]`, `backend/validation/retuning_tier1.py:45`). It is not shipped, and the source project skips these tests too.

The BO tests gave 17 passed. The slow set gave **11 passed and one known, pre-existing failure**: the golden digest `payload/validate.excitation/__digest` (285/286 golden values match). That failure predates this work; see REVIEW_GUIDE §7 for what the digest covers.

**Note.** The tests write real retune results to `data/session/cache/`. Delete `data/session/` before the dashboard check, or the first retune can be a cache hit instead of a live ~90 s run.

## Start the dashboard

```bash
BACKEND_PORT=8044 FRONTEND_PORT=5318 ./start_dashboard.sh
# Dashboard http://127.0.0.1:5318/   API docs http://127.0.0.1:8044/docs   health /health
```

- The defaults are 8024/5298. The script refuses to start if a port is busy.
- It expects `.venv/` at the bundle root and `frontend/node_modules/` from `npm ci`.
- It writes logs to `logs/backend.log` and `logs/frontend.log`.
- To retune: open the **Twin Study** tab, pick a plant and drift, then press **1 · RUN PHYSICAL MACHINE → 2 · OPTIMISE GAIN → 3 · APPLY TO THE MACHINE**.
- Runtime state goes to `data/session/`: the result cache in `data/session/cache/` and the saved six gains in `data/session/zone_gains/`. Delete that folder to start clean.
- The Validation screens answer from the shipped caches in `reports/validation_summary/` when the backend runs from the bundle root, which `start_dashboard.sh` does. The Excitation cache records its files as bundle-relative paths, so a backend started from another directory misses that cache and recomputes the study (about 9 min on CPU).

## Folder map

| path | what |
|---|---|
| `backend/pipeline/retune.py` | Algorithm 1 steps 4–7 live: cost tier, gain structures, two-stage gains, HGS calls, step-7 restore |
| `backend/pipeline/zone_gains.py` | store for the saved six per-zone gains (per plant and cost tier) |
| `backend/validation/retuning_tiered_eval.py` | cost tiers T0–T3 (T1 = the repaired cost) |
| `backend/validation/retuning_paper_eval.py` | Eq. (12) evaluator (T0), the authors' arithmetic |
| `backend/validation/retuning_paper_protocol.py` | the authors' HGS (2,805 evals), search box, BO call (`run_bo`) |
| `backend/validation/retuning_structure_search.py` | dimension-generic search used for the 6-D (six-gain) stage |
| `backend/validation/retuning_structures.py` | gain structures: shared-2D, outfeeder-only-2D, ends-only-2D, ends-tied-4D, full-6D |
| `backend/api/session_routes.py` | `/retune`, `/zone-gains`, `/zone-gains/delete`, `/plant/step`, `/plant/baseline`, … |
| `backend/pipeline/precompute.py` | offline precompute of dashboard cases (code only; its output is not shipped) |
| `frontend/` | React/Vite dashboard (`src/tabs/TwinStudy.jsx` is the retune screen) |
| `tests/` | backend test suite (37 files); `conftest.py` isolates the zone-gains store |
| `checks/` | golden-value regression probes (`golden_values.json`) |
| `data/paper_reference/`, `data/gain_atlas/`, `paper_package/` | the paper's plant/excitation tables and reference values the backend reads |
| `reports/section4_author_spec/`, `reports/validation_summary/`, `reports/figures/` | precomputed cells, the caches the Validation screens serve, and the few figures and CSVs those caches must find to be accepted |
| `reports/section4_tier2_twinfix/cells/`, `reports/bo_config_study/`, `reports/section4_tier2_{cpu,record,wos0}/` | the pre-reply campaign, the BO-configuration study and three diagnostic campaigns. `/validate/retuning` reports them as superseded or diagnostic, and `tests/test_retuning_bo_sensitivity.py` and the retuning golden digest need them |
| `bo_campaign/` | the out-feeder BO campaign: `src/`, `tests/`, `reports/` (spec, results, 60 cells, commissioning), `tables/`, `figures/`. See `bo_campaign/README.md` |
| `docs/reports/` | background reports: RETUNING-SUMMARY, FINAL-REPORT, HOW-IT-WORKS, KP-TI-EXPLAINED, SEQUENTIAL-HGS (md + pdf). Code comments cite them as `multiloop/reports/…` (see REVIEW_GUIDE §5, "Names in code comments") |
| `docs/plans/2026-10-07-t1-two-stage-bo.md`, `docs/DECISIONS-LEDGER.md` | the implementation plan and the ruling-by-ruling ledger of the 7–8 Oct work |
| `run_gain_atlas.py`, `run_full_sweep.py` | offline generators of the gain atlas and sweep the dashboard serves (the tests import them) |


