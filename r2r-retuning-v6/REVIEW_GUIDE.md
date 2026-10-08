# Review guide — r2r-retuning-v6

*For a senior cross-check, 8 Oct 2026. Every path, function and line number below was checked with `grep -n` against this bundle. Every number comes from a file in the bundle, and the file is named. All results are from simulation.*

**Plant names.** The paper's retuning plants P001, P049, P053, P158, P186, P189 are the dashboard's P01, P02, P03, P06, P09, P10. A *cell* is one plant with one drift (D01–D10), so 6 × 10 = 60 cells. The *twin* is the model identified from the drifted line's logged data. *HGS* is the authors' hierarchical grid search.

---

## 1. What the paper does, what changed, and why

**The paper (Algorithm 1, steps 4–7).** When drift is suspected:
- identify θ̂ from a logged excitation and build the twin;
- search the twin for **one shared gain pair** (K_p\*, T_I), used on all three tension zones (UW unwinder, out-feeder nip, RW rewinder; the in-feeder is the speed master and has no tension gain);
- score the pair with Eq. (12) on a +20 % step in all three tension setpoints;
- compare the result with C_target, the pre-drift cost.

The search is HGS over K_p\* ∈ [1, 500] (log scale) and T_I ∈ [0.5, 30] s (absolute). The optional few-shot BO on the line is HGS+BO(5).

The bundle changes four things. A fifth item fixes the reference that results are measured against. Each change was the user's decision, recorded in `docs/DECISIONS-LEDGER.md`.

### 1a. Cost: T1 replaces Eq. (12) as the default; T0 is still selectable

**T0 = Eq. (12)**, as transcribed from the authors' reply (the reply itself is not shipped; see README, "Deliberately not included"). It is scored over the 25 s after the step at 5 s, with RK4 at dt 1 ms. The band is b_i = 2 % of span i's step.

> S_T0 = (mean_i RMSE_i / 3 N)² + 2·(max_i OS_i / 20 %)² + (t_s / 3 s)²
>
> t_s = the last time any span is outside ±b_i

**T1 = the repaired cost.** It uses the same experiment and the same dynamics. Only the cost changes:

> e_i = RMSE_i / |T_ref,i| (post-step setpoint) OS_i = 100·max(err_i)⁺ / |ΔT_i| t_out,i = Δt · Σ_t σ((|err_i(t)| − b_i) / (0.15·b_i))
>
> J_i = (e_i / 0.02)² + 2·(OS_i / 20 %)² + (t_out,i / 3 s)²
>
> **S_T1 = ( (1/3) Σ_i J_i⁴ )^(1/4)**

T1 repairs three defects in Eq. (12):

| Eq. (12) defect | T1 repair | evidence in the bundle |
|---|---|---|
| Error is divided by a fixed 3 N, but setpoints run 12–360 N on the six plants. So 3 N is 25 % of setpoint on P01 and 0.83 % on P09/P10, a 30× difference in strictness. | Error is taken relative to each span's own setpoint. | `retuning_tiered_eval.py` docstring; `docs/reports/RETUNING-SUMMARY.md` §2 |
| Settling time is the *last* sample outside the band. It jumps by a whole oscillation when a wobble grazes the band. | Total time outside the band, smoothed by a sigmoid, so it is continuous in the gains. | On P001/D07, a 1.14 % K_p\* change (3.6634 → 3.7050): T0 goes 0.16514 → 0.28402 (**+72.0 %**); T1 goes 1.53926 → 1.50051 (**−2.5 %**). Printed by `tests/test_tiered_eval_matches_study.py::test_t1_removes_the_settling_cliff`. |
| The spans are combined three ways: mean for RMSE, max for overshoot, any for settling. With per-zone gains, the mean can hide one bad zone behind two good ones. | One complete J per span, aggregated once with a 4-norm (close to the worst span, still smooth). | design argument; not isolated by a test |

T2 adds drive limits, noise, a filter and an effort term on top of T1. T3 adds three roll positions. Both are selectable but not used as defaults.

**T0 is still bit-identical.**
- `_Scorer` at T0 routes to the same `PaperEvaluator` the pipeline used before tiers existed.
- `TieredEvaluator` at T0 delegates to `PerRollerEvaluator`, a per-roller kernel pinned to `PaperEvaluator`. There is no second copy of Eq. (12).
- The tests assert `== 0.0`, not `allclose`.

What "bit-identical" means here: identical to `PaperEvaluator`, the project's own transcription of Eq. (12) as used before cost tiers existed. It is not a comparison with the authors' code, which is not available. The link to the authors is assumption A3 (`retune.py:128`): they confirmed that an independent transcription matches their costs to a median ratio of 1.0015.

### 1b. Per-zone gains, two stages (default `gain_structure = "outfeeder-only-2D"`)

**Why per-zone.** The zones do different jobs. Given their own gains they choose different K_p\*. Per-case medians: UW 16.8, nip 11.1, RW 39.4 s⁻¹. RW is higher than UW in 59 of 60 cases, although the two are identical machines (`RETUNING-SUMMARY.md` §4).

**The two stages.**
1. **Stage 1, `full-6D`, once per plant.** The 6-D structure search (§2) finds a free (K_p\*, T_I) per zone on the twin. The six gains are **saved** in `zone_gains`, keyed by **plant + cost tier**; drift and protocol are not part of the key.
2. **Stage 2, `outfeeder-only-2D`, every later case.** UW and RW are **frozen** at the saved values. The authors' 2-D HGS retunes only the out-feeder, so the search space is 2-D, as in the paper.

Three further rules:
- If nothing is saved yet, stage 2 runs stage 1 first.
- The six gains are saved **only if step 7 did not restore** (`_commit_per_zone`, `retune.py:813`): under A9, the twin's gains must beat today's gains on the drifted line. The C_target pass/fail does not matter here, so a case that improves the line but "fails" C_target still commissions the plant. A restored case does not.
- The result cache key includes the frozen ends, so a cached answer never outlives a change to the saved six.

**Frozen UW/RW values used by the BO campaign.** These come from `bo_campaign/reports/bo_outfeeder_cells/commissioning/<plant>.json`: the 6-D twin search on the **pre-drift** twin, T1, seed 0, 12,175 twin evaluations each. The out-feeder pair is listed only as its commissioning value; it is retuned in every cell.

| plant (dash) | UW K_p\* | UW T_I s | RW K_p\* | RW T_I s | out-feeder K_p\* / T_I (commissioning) | S_twin | S pre-drift plant | twin MARE % |
|---|---|---|---|---|---|---|---|---|
| P001 (P01) | 8.024 | 23.156 | 19.816 | 10.519 | 4.085 / 30.000 | 0.53594 | 0.54899 | 3.61 |
| P049 (P02) | 9.526 | 14.662 | 36.639 | 20.470 | 8.364 / 23.374 | 0.22309 | 0.21878 | 7.61 |
| P053 (P03) | 19.615 | 12.345 | 40.062 | 29.905 | 17.874 / 16.424 | 0.09716 | 0.09707 | 12.01 |
| P158 (P06) | 49.795 | 13.791 | 122.169 | 9.229 | 35.248 / 22.528 | 0.04421 | 0.04317 | 12.41 |
| P186 (P09) | 35.953 | 24.654 | 129.173 | 10.584 | 37.056 / 18.777 | 0.04733 | 0.04640 | 19.23 |
| P189 (P10) | 10.596 | 15.592 | 15.687 | 26.400 | 6.016 / 18.595 | 0.27304 | 0.38608 | 285.04 |

The dashboard does not read these files. It saves its own six gains from the twin of the **first case of each plant that step 7 did not restore**. The campaign used the pre-drift twin instead, so that every cell freezes the same ends regardless of case order.

### 1c. Step-7 restore (assumption A9; the paper leaves this open)

The paper says that on failure you return to step 3 with a richer excitation. It does not say which gains run in the meantime.

**A9.** The twin's gains are delivered only if their S on the drifted line is **strictly below** the S of the commissioned gains on the same drifted line. Otherwise the commissioned gains are restored. C_target is still checked and reported on whatever is delivered.

The rule handles both failure modes:
- Restoring on the C_target test alone would undo a retune that helps but cannot reach the pre-drift cost. On P06 a retune cut the cost by 23.9 % and still "failed" C_target (`RETUNING-SUMMARY.md` §2).
- Never restoring would ship worse gains when θ̂ is badly wrong (P10: twin MARE 285.04 % in `commissioning/P189.json`; the A9 text in `retune.py:148` rounds it to "~280 %").

The restore is pure and idempotent. It is applied on the way out, cache hits included. The proposal stays visible as `gains.twin_candidate`.

### 1d. "Best possible" reference = the same structure on the true plant

The screen's "Best possible (same gains fixed)" value and `ratio_vs_reference` come from the same structure searched on the true drifted plant. For the default, that means HGS on the out-feeder with UW and RW at the saved six; this is the BO campaign's *floor*. It used to be the paper's shared-pair optimum, against which per-zone results looked about 36 % "better than best". Nothing on a real line could compute this reference; it is a simulation yardstick.

---

## 2. Where each item lives (bundle paths, verified line numbers)

| item | file : line — symbol |
|---|---|
| default cost tier | `backend/pipeline/retune.py:73` `DEFAULT_COST_TIER = "T1"` |
| T0 routes to Eq. (12) | `retune.py:273` `class _Scorer`; `:289` `PaperEvaluator(...) if tier == "T0"` |
| Eq. (12) formula | `backend/validation/retuning_paper_eval.py:128` (`S = (rmse / S_E_N) ** 2 + ...`); `:155` `class PaperEvaluator` |
| T1 constants | `backend/validation/retuning_tiered_eval.py:73–76` `S_E_REL`, `S_T_OUT`, `SOFT_BAND`, `P_NORM` |
| T1 formula | `retuning_tiered_eval.py:114` `_make_cost`; `:227` band; `:235` `err_rel`; `:236–237` sigmoid `t_out_span`; `:239` `j_span`; `:242` 4-norm `S` |
| T0 inside the tiered evaluator | `retuning_tiered_eval.py:270` `class TieredEvaluator`; `:286–290` delegates to `PerRollerEvaluator` |
| structures | `retune.py:88` `PER_ZONE_STRUCTURE = "full-6D"`; `:89` `OUTFEEDER_STRUCTURE`; `:90` `DEFAULT_GAIN_STRUCTURE`; `backend/validation/retuning_structures.py:97–109` |
| two-stage flow | `retune.py:471` `retune`; `:475` loads saved six; `:556` stage 1 `structure_search(... full-6D)`; `:581` stage 2 `_hgs_one_zone(twin_scorer, kp6, ti6, zone=1)` |
| cache key with frozen ends | `retune.py:458` `_retune_key`; `:463` `content["frozen"]` |
| save only if step 7 did not restore | `retune.py:784` call; `:797` `_commit_per_zone`; `:813` the restore branch; `:823` `_save_per_zone` |
| few-shot BO in the dashboard | `retune.py:684–686`: under any per-zone structure (the default included), `bo_refine` returns `not_applicable`. HGS+BO is evaluated only in `bo_campaign/`. `BO-OUTFEEDER-RESULTS.md` §7 recommends HGS-only as the default, with HGS+BO(5) or (10) as an optional "refine on the line" step that is not wired yet |
| zone-gains store | `backend/pipeline/zone_gains.py:25` `STORE_DIR` (= `data/session/zone_gains`); `:28` `key` (plant + tier); `:37` `load`; `:51` `save` (atomic temp + `os.replace`); `:76` `delete` |
| step 7 / A9 | `retune.py:146` A9 text in `ASSUMPTIONS`; `:730` step 7; `:829` `apply_restore`; `:845` the rule `restored = ...` |
| "best possible" | `retune.py:641` out-feeder HGS on the true plant (default); `:645` 6-D on the true plant (`full-6D`); `:650` shared HGS (others) |
| "twin, gains today" | `retune.py:661` `commissioned["S_twin"]` |
| HGS | `backend/validation/retuning_paper_protocol.py:137` `hierarchical_grid_search` (see §3) |
| 6-D structure search | `backend/validation/retuning_structure_search.py:71` `search`; `:22` `GLOBAL_PER_DIM = 1500`; `:97` `n_global` |
| routes | `backend/api/session_routes.py:442` `POST /retune`; `:501` `POST /zone-gains`; `:509` `POST /zone-gains/delete`; `:517` `POST /plant/baseline`; `:528` `POST /plant/step`; `:103` `RetuneBody` (defaults read from the pipeline) |
| retune screen | `frontend/src/tabs/TwinStudy.jsx` (buttons at `:580`, `:674`; "FORGET SAVED SIX GAINS" at `:669`; "Best possible" at `:631`) |

**6-D structure search.** It is dimension-generic because a 30^6 grid does not exist. It runs four stages:
1. global LHS of 1,500 × d points;
2. zoom 1 of 25 % of that, in a window of ±25 % of the range;
3. zoom 2 of 10 %, in a window of ±6.25 %;
4. a coordinate polish of 1 + 4·d points.

For d = 6 that is 9,000 + 2,250 + 900 + 25 = **12,175** evaluations, which matches `twin_evals` in every commissioning file.

---

## 3. HGS — the authors' search, and where it is used

**Definition.** `retuning_paper_protocol.py:137` `hierarchical_grid_search(batch_costs, *, seed=0)`. The box and stages:
- **Box:** `:32` `KP_BOUNDS = (1.0, 500.0)`, log; `:33` `TI_BOUNDS_S = (0.5, 30.0)` s, absolute and linear. These come from the authors' reply (assumption A1, `retune.py:122`).
- **Stages:** `:39` `HGS_STAGES = {"coarse_side": 30, "lhs": 1000, "fine_side": 30, "polish": 5}`. The paper (S8.1) gives the total of 2,805 twin evaluations and names the stages (the code comment at `:37–38`). The split 900 / 1,000 / 900 / 5 and the spacing are the dashboard's documented choice, because neither the paper nor the reply gives them (A4, `retune.py:131`).

| stage | lines | what | evals |
|---|---|---|---|
| coarse | `:161–163` | 30 × 30 grid; K_p\* `geomspace(1, 500)`, T_I `linspace(0.5, 30)` | 900 |
| LHS | `:165–167` | 1,000-point Latin hypercube (`seed=0`), K_p\* log-mapped | 1,000 |
| fine | `:169–175` | 30 × 30 around the incumbent: K_p\* in [k/2, 2k] (log), T_I ± 12.5 % of the range (±3.69 s), clipped | 900 |
| polish | `:177–179` | 5 points: K_p\* ×1.05 and ×0.95, T_I ×1.05 and ×0.95, both ×1.02 | 5 |
| **total** | | deterministic | **2,805** |

`retune.py:105` `HGS_BUDGET` recomputes the same sum.

**Where the dashboard uses HGS** (`backend/pipeline/retune.py`):
- `:532` **commissioning.** HGS on the *true pre-drift* plant gives the shared pair, which is "gains today" and the step-7 restore target. Its cost becomes C_target (`:534`).
- `:581` **stage 2.** HGS on the out-feeder of the twin, with UW and RW at the saved six, through `_hgs_one_zone` (`:416`).
- `:641` **"best possible".** The same `_hgs_one_zone`, on the true drifted plant.
- `:591` `shared-2D` (the paper's own HGS), `:600` the baseline for restricted structures, and `:445` `sequential-2D`.

**Keep-incumbent guard** (`_hgs_one_zone`, `retune.py:425–435`). HGS does not seed its grid with the current gains, so it can return a grid point marginally *worse* than where the zone already was. A live case went 0.15203 → 0.15302 before the guard existed. The guard scores the incumbent and keeps it unless HGS is strictly better (`:431` `if not (float(h.cost) < current)`). A stage therefore never raises the twin cost. `search.outfeeder.kept_saved_value` reports when the guard fired.

**Where the BO campaign uses HGS.** `bo_campaign/src/run_paper_methods.py:108` `_twin_search` calls `AP.hierarchical_grid_search` (`:115`) for any 2-D structure. `run_bo_outfeeder.py` calls it three times:
- `:183` the **HGS-only** arm, on the twin;
- the same answer is the warm-start centre of **HGS+BO(5)/(10)**;
- `:204` the **floor**, on the true plant (the `seed=7` there only affects the 6-D search; HGS's LHS is always seed 0).

Every cell records `twin_evals = 2805` for HGS-only. **The campaign's HGS-only arm is plain HGS without the guard**, as in the paper. This is a deliberate ruling in the ledger, and the difference is tiny.

`run_paper_methods.py` also has its own BO (`_bo`, `:77`; the warm start at `:92`), used by the 6 Oct `paper_cells/`. It warm-starts with `keep = max(1, budget // 2)` star points, so its WS-BO(30) used about 15 seeds. The split table in `bo_campaign/README.md` (5 / 8 / 17 for WS-BO(30)) is `run_bo_outfeeder.bo_split` and applies only to the final campaign. From `paper_cells/` this campaign uses only the 6-D floor and the reproducibility gate.

---

## 4. The BO campaign, in brief

- **Setup.** 60 cells, cost T1, scored on the true drifted plant. UW and RW are frozen at the commissioning values above, and only the out-feeder's (K_p\*, T_I) is searched.
- **Arms:**
  - *doing nothing* (`commissioned-6`);
  - HGS-only, with 0 line runs;
  - HGS+BO(5) and HGS+BO(10), seed 0;
  - CS-BO(30) and WS-BO(30), seeds 0, 1 and 2.
- **Floor:** HGS on the true plant.
- **Hardware:** GPU (RTX 5070 Ti); every cell records `device: cuda:0`.

`bo_campaign/README.md` gives the exact `gp_minimize` call, the seed/random/EI split, the re-run instructions and the reproducibility gate. The full spec is `bo_campaign/reports/BO-OUTFEEDER-SPEC.md` and the results are in `bo_campaign/reports/BO-OUTFEEDER-RESULTS.md`.

---

## 5. How to cross-verify

Run everything from the bundle root after the install in `README.md`.

**Test suite.**
```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q                       # fast set: 430 passed, 66 skipped, ~10 min on CPU
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q -m slow               # 11 passed + 1 known failure (§7)
(cd bo_campaign && JAX_PLATFORMS=cpu ../.venv/bin/python -m pytest -q tests)   # 17 passed
```

**Tests that pin each claim.**

| claim | tests |
|---|---|
| T1 is the default; tier in cache key; bad tier rejected | `tests/test_retune_cost_tier.py` |
| T0 bit-identical (`== 0.0`) to `PaperEvaluator` and the authors' kernel | `tests/test_retune_cost_tier.py::test_t0_scorer_is_bit_identical_to_the_paper_evaluator`, `tests/test_tiered_eval_matches_study.py::test_t0_still_reproduces_the_authors_kernel` |
| backend T0–T3 = the campaign's copy (`bo_campaign/src/tiers.py`), 0 difference | `tests/test_tiered_eval_matches_study.py::test_backend_matches_the_study_copy` |
| T1 removes the settling cliff | `tests/test_tiered_eval_matches_study.py::test_t1_removes_the_settling_cliff` |
| Eq. (12) evaluator and protocol (split, star, shared random points, HGS = 2,805) | `tests/test_retuning_paper_protocol.py`, `tests/test_retuning_campaign.py` |
| two-stage: default, save, reuse, frozen ends, cache key, "best possible", commissioned S_twin | `tests/test_retune_two_stage.py` |
| six gains saved only when step 7 does not restore; re-commit after forget | `tests/test_retune_two_stage_commit.py` |
| zone-gains store (key, round trip, concurrency, corrupt record) | `tests/test_zone_gains_store.py` |
| `/zone-gains`, `/zone-gains/delete` | `tests/test_zone_gains_route.py` |
| `/retune` wiring, 422/400 contracts, end-to-end (slow) | `tests/test_retune_route.py` |
| step-7 restore A9 (worse → restore, tie → keep, divergent → restore, idempotent, cache hit) | `tests/test_retune_restore.py` |
| keep-incumbent guard (twin cost never rises stage to stage) | `tests/test_retune_sequential.py::test_exactly_the_published_procedure`; it exercises `_hgs_one_zone` through `sequential-2D` |
| per-zone structures, scalar path unchanged | `tests/test_retune_per_zone_structures.py`, `tests/test_per_zone_gains.py` |
| BO spec (split matches what skopt actually does, star, seed designs, GPU guard) | `bo_campaign/tests/test_bo_outfeeder_spec.py` |
| aggregation (gap, pooling, below-floor counted) | `bo_campaign/tests/test_aggregate_bo_outfeeder.py` |

`/plant/step` and `/plant/baseline` have **no dedicated test**. The UI exercises them, but no test pins them.

**Names in code comments.** Some comments and docstrings still use the source project's names:
- `multiloop/src/…` is `bo_campaign/src/…` here, and `multiloop/reports/X.md|pdf` is `docs/reports/X.*` (or `bo_campaign/reports/` for the BO-OUTFEEDER files).
- `tests/test_t0_identity.py` (cited in the `retuning_tiered_eval.py` docstring) is not shipped. The same T0 identity is pinned by `tests/test_tiered_eval_matches_study.py` and `tests/test_retune_cost_tier.py`.
- "delegates to paper_t0" (`retuning_tiered_eval.py:232`): in the backend, T0 delegates to `retuning_per_zone_eval.PerRollerEvaluator`. `paper_t0` is the campaign's copy (`bo_campaign/src/paper_t0.py`).
- The docstring's settling-cliff figure, "a 0.14 % gain change … cost +67 %" (`retuning_tiered_eval.py:17–18`), is an older measurement. The current test prints a 1.14 % change and +72.0 % (§1a).

**Re-run one BO cell and compare** (GPU required; a few minutes alone on an RTX 5070 Ti, where the gate probe took 227 s). The runner skips any cell whose JSON already exists, so move the stored copy aside first:
```bash
cd bo_campaign
export LD_LIBRARY_PATH="$(ls -d "$PWD"/../.venv/lib/python3*/site-packages/nvidia/*/lib | tr '\n' ':')" XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p /tmp/bo_ref && mv reports/bo_outfeeder_cells/P001__D01.json /tmp/bo_ref/
../.venv/bin/python src/run_bo_outfeeder.py --only P001/D01
../.venv/bin/python - <<'EOF'
import json
a = json.load(open("/tmp/bo_ref/P001__D01.json")); b = json.load(open("reports/bo_outfeeder_cells/P001__D01.json"))
for arm in a["methods"]:
    print(f"{arm:15s}", [round(r["S_plant"], 6) for r in a["methods"][arm]], [round(r["S_plant"], 6) for r in b["methods"][arm]])
print("floor", a["floor"]["S_plant"], b["floor"]["S_plant"])
EOF
```
The stored values, for reference: HGS-only 0.519031, CS-BO(30) [0.519274, 0.519407, 0.522327], floor 0.5190230. The CPU-vs-GPU gate is described in `bo_campaign/README.md`.

**Start the dashboard and do one retune.**
1. Start it with `BACKEND_PORT=8044 FRONTEND_PORT=5318 ./start_dashboard.sh` and open **Twin Study**.
2. Pick a plant and drift, then press 1 · RUN PHYSICAL MACHINE → 2 · OPTIMISE GAIN → 3 · APPLY TO THE MACHINE.
3. On the first case of a plant, the six gains are searched and, unless step 7 restores today's gains, saved. The next drift reuses them (6-D evaluations 0). FORGET SAVED SIX GAINS clears them.

If you ran the test suite first, delete `data/session/` before this check. The tests leave retune results in `data/session/cache/`, and a cache hit replaces the live run.

Without the UI:
```bash
curl -s -X POST http://127.0.0.1:8044/retune -H 'Content-Type: application/json' -d '{"run_spec":
 {"plant":{"source":"preset","preset_id":"P01"},
  "drift":{"EA_pct":0,"J_UW_pct":-30,"J_Nip_pct":0,"J_RW_pct":50,"f_pct":0},
  "protocol":{"T_log_ms":5,"excitation":"E_Toggle","record_s":16,"pct_T":0.003,"pct_v":0.003,
              "LPF_T_hz":50,"LPF_v_hz":50,"Kp_star":100,"seed":0}}}' \
 | python -c "import json,sys; r=json.load(sys.stdin); print(r['cost_tier'], r['options']['gain_structure'], r['search']['per_zone_gains']['reused'], r['restore'], r['gains']['delivered']['kp_star_per_zone'], r['ratio_vs_reference'])"
```
- Add `"cost_tier":"T0","gain_structure":"shared-2D"` to the body to get the paper's own method.
- `POST /zone-gains` with the same `run_spec` shows the saved six.
- On CPU a first-case retune took 87 s in a check of this bundle (the 6-D search is 12,175 twin evaluations). That check printed: `T1 outfeeder-only-2D False {'restored': False, 'S_candidate': 0.5494, 'S_commissioned': 0.9070, 'rule': 'A9'} [8.024, 4.172, 19.816] 1.0004` (values rounded here).
- The saved six from this case (UW 8.024 / T_I 23.156 s, RW 19.816 / 10.519 s) equal the campaign's `commissioning/P001.json`, although the dashboard never reads that file and this twin is a drifted one. The reason: the 6-D search draws a fixed seed-0 sample set, and nearby twins pick the same winning sample. The twin costs differ (S_twin 0.5344 here, 0.5359 in the commissioning file).
- Step 5 (twin validation) can report "fail" without stopping the retune: it is reported, and only step 7 decides what is delivered (`retune.py:708–727`). A smoke check of this bundle hit this on a second P01 drift (twin RMSE 0.00146 N against ε 0.00111 N); step 7 still passed.
- The response fields `retune_version` "v5" and `step_version` "v3" are cache-format versions (`retune.py:65`, `backend/pipeline/step.py:33`), not the V6 label.

---

## 6. Headline results (from the bundle's reports)

**T0 reproduces the paper.** `reports/section4_author_spec/scorecard.md` covers 120 cells with 0 failures. Of the paper's Section 4 / Table 3 items, it marks 80 REPRODUCED, 9 CLOSE and 6 DIFFERS. For example, CS-BO(30) median S is 0.4074 against 0.407 in the paper, and HGS-only median is 0.3477 against 0.357.

**Per-zone gains (6 Oct, `sequential-2D`, T1, 60 cells; `docs/reports/RETUNING-SUMMARY.md` §4).**
- Median cost change against the commissioned shared pair: **−42.2 %** (worst −30.9 %).
- Against the paper's retuned shared pair: **−40.8 %**. Every case improved.
- Six free gains beat the paper's retuned pair by a median 36.2 % even under Eq. (12).

`sequential-2D` is the 6 Oct method; the current default (two-stage) was adopted on 7 Oct.

**Out-feeder BO campaign (two-stage, T1; `bo_campaign/tables/bo_outfeeder_summary.csv`).** Gap = S / S_floor − 1.

| arm | line runs | runs | median gap % | 75th pct % | worst % | median vs doing nothing % | worse than doing nothing |
|---|---|---|---|---|---|---|---|
| doing nothing | 0 | 60 | 1.632 | 4.881 | 83.145 | — | — |
| HGS-only | 0 | 60 | 0.062 | 0.393 | 22.531 | −1.527 | 7 |
| HGS+BO(5) | 5 | 60 | 0.060 | 0.366 | 3.435 | −1.527 | 7 |
| HGS+BO(10) | 10 | 60 | 0.056 | 0.304 | 3.244 | −1.543 | 7 |
| CS-BO(30) | 30 | 180 | 0.081 | 0.381 | 7.519 | −1.215 | 27 |
| WS-BO(30) | 30 | 180 | 0.169 | 0.436 | 7.574 | −1.245 | 26 |

**Price of freezing UW and RW at commissioning.** Measured as floor / six-gain floor − 1 over the 60 cells (`BO-OUTFEEDER-RESULTS.md` §5):
- median **+2.11 %**; per-cell maximum +423 % (P186 D07);
- per-plant medians: P001 0.48, P049 2.28, P053 3.83, P158 0.05, P186 3.58, **P189 21.8 %**;
- the price exceeds what HGS-only recovers in 36 of 60 cells. "Recovers" here means HGS-only's cost reduction relative to doing nothing, (1 − S_HGS-only / S_doing-nothing) × 100. Other definitions give 34 (S_doing-nothing / S_HGS-only − 1, or the difference of the two gaps to the floor). `aggregate_bo_outfeeder.py` does not compute this count; it was recomputed from the 60 cells.

**Step-7 restore (A9).** On 7 Oct it restored 6 of 198 cached retunes, all T0 (`RETUNING-SUMMARY.md` §5).

---

## 7. Known limitations and open decisions

- **Pre-existing golden drift (do not "fix").** The slow set fails exactly one probe, `payload/validate.excitation/__digest`; 285 of 286 golden values match.
  - What it covers: `checks/probes.py` posts `/validate/excitation` with `{"plant_id": "ALL"}`, flattens every numeric leaf of the response (paths, URLs and timestamps excluded), and hashes the canonical list with SHA-256 (`digest_leaves`, `checks/probes.py:229`).
  - Values: stored 2680978347757801 (`checks/golden_values.json:206`), current 2591239530530286. The leaf count (3951) still matches, so the structure is the same and some numbers differ.
  - Why it is not from this work: the excitation study does not touch retune, the cost tiers or the gains. The source project's progress notes (not shipped) record it as "proven pre-existing by a swap test against V5's file" (V5 is the earlier project this one was copied from). The ledger (`docs/DECISIONS-LEDGER.md:38`) records 285/286 unchanged before and after the 7–8 Oct work.
- **Test counts.** The fast set gives 430 passed / 66 skipped / 0 failed, both in this bundle and in the source project. Every skip depends on the environment: 22 in `tests/test_retuning_jax.py` need `JAX_ENABLE_X64=1`, and 44 in `tests/test_retuning_tier1.py` need the external v5 figure package (README, "Deliberately not included").
- **Identification bias on fast plants (P05/P07/P08/P10) matches the paper.** On P10 the fitted parameters are about 285 % off; the paper reports about 293 % for P189. The cause was found: velocity-log noise leaks into the tension fit and under-estimates web stiffness. It was deliberately not fixed, to keep the paper's excitation (`RETUNING-SUMMARY.md` §5). In the BO campaign this is the source of P189's tail.
- **"Gains today" is still the paper's shared pair.** It is also the restore target. The pair is HGS on the pre-drift plant (`retune.py:532`), not the saved six. This is an open user decision; the ledger marks it "ASK USER". Using the saved six would restore 7 of 60 HGS-only cells and move the median 0.062 → 0.057 % (`BO-OUTFEEDER-RESULTS.md` §7 (a)).
- **Precompute not included.** The cached dashboard cases were stopped to give the BO campaign the GPU and were not relaunched. `backend/pipeline/precompute.py:38` still names `sequential-2D`. Relaunching under the new defaults is decision (b) in `BO-OUTFEEDER-RESULTS.md` §7. Until then every click computes live.
- **Older reports predate 7 Oct.** `docs/reports/RETUNING-SUMMARY.md` §3 says the dashboard runs `sequential-2D` under T0. That was true when the report was written, before the 7 Oct switch, and is superseded. **Where a report and the code disagree, the code (`retune.py:73`, `:90`) is current.**
- **Search box.** The out-feeder T_I sits on the 30 s ceiling in 42 of 60 floors of the BO campaign (T1, `BO-OUTFEEDER-RESULTS.md`). This is a different population from A1's "59 of 60 twin optima" (`retune.py:124`): that is the authors' own count for the paper's shared-pair search under Eq. (12), as reported in their reply. The dashboard's reproduction of that campaign puts at most 47 of 60 there. The authors note the truncation. The SysID T_I(θ̂) of 36–50 s is clipped for WS-BO.
- **Statistics.** There is one commissioning point per plant, and HGS+BO is single-seed. Everything is simulation, and the T2/T3 drive and roll parameters are assumptions.
- **Deferred minor items** are listed at the end of `docs/DECISIONS-LEDGER.md`. Examples: the FORGET button has no confirm, and the "Best possible" wording is wrong for `full-6D`.
