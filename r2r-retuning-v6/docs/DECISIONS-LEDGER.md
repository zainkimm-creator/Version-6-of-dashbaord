# SDD ledger — plan: docs/plans/2026-10-07-t1-two-stage-bo.md
Setup: Ruling: no git repo — no worktree/commits; ledger by hand; backups of touched files in job tmp — cost if wrong: none (user owns history)
Setup: Ruling: run_phase5b.py (named in the user spec) is not on this machine; run_paper_methods._bo reproduces the spec split and the seed-0 random designs exactly (verified) — cost if wrong: re-run with the original file
Setup: Ruling: precompute stopped with user approval (AskUserQuestion) to give BO the GPU — relaunch under new defaults in Task 8
Pre-flight: Task 4 consumes Task 3 store API; Task 6 consumes Task 4 payload search.per_zone_gains + Task 5 routes; Task 1 independent of 2-6 (uses backend/validation only)
Ruling: BO campaign freezes UW/RW at the 6-D optimum of the PRE-DRIFT twin (commissioning), not per drift case — mirrors "save 6 gains, reuse for the next case" order-independently — cost if wrong: re-run (per-case variant already exists on CPU in paper_cells for comparison)
Task 1: launched BO campaign 23:40 (6 workers, GPU); spec tests 11/11 pass (tests/test_bo_outfeeder_spec.py); GPU probe P001/D01 identical to CPU paper_cells to 6 dp, 227 s/cell
Task 1: Ruling: parallel commissioning OOMed (6 x 6-D on 16 GB) for P001/P049/P158; commissioned them sequentially, pass-2 workers (2) for the 20 errored P001/P049 cells; launcher made sequential + stderr logged — cost: ~20 min
Task 2: complete (DEFAULT_COST_TIER T1; tests/test_retune_cost_tier.py 8/8; full suite deferred to Task 5 gate)
Task 2: Ruling: full-suite run batched with Tasks 3-5 (5 min each) — cost if wrong: a regression found later, attributed by file
Task 4: Ruling: fixtures in test_retune_per_zone_structures.py:23 and test_retune_sequential.py:27 used RetuneOptions() to mean the shared pair; now name shared-2D explicitly — the default moved by spec — cost if wrong: none
Task 3: complete (backend/pipeline/zone_gains.py; tests/test_zone_gains_store.py 6/6 RED->GREEN)
Task 4: complete (two-stage in retune.py: PER_ZONE/OUTFEEDER structures, store save/reuse, key incl. frozen ends; tests/test_retune_two_stage.py 7/7 incl. commissioned S_twin RED(KeyError)->GREEN)
Task 4: Ruling: out-feeder stage uses _hgs_one_zone (authors' HGS + keep-incumbent guard), so the dashboard's out-feeder never ends worse on the twin than the saved value; the BO campaign's HGS-only arm is plain HGS (as the paper) — cost if wrong: tiny dashboard/campaign difference when HGS lands worse than the saved nip
Task 4: Ruling: commissioned gains (step 1 baseline, restore reference) stay the paper's shared commissioning pair, not the saved six — keeps A9 and the "today" baseline unchanged — cost if wrong: "gains today" panel shows the shared pair even after the six were applied
Task 5: complete (POST /zone-gains, /zone-gains/delete; tests/test_zone_gains_route.py 4/4 RED(404)->GREEN)
Task 6: in progress — Ruling: no frontend unit-test framework; verified by vite build + live screenshots. Fixed display bug: column 3 showed the proposal, now step-7 delivered; "twin, gains today" showed the LINE cost, now commissioned.S_twin (new backend field)
Task 7: aggregate_bo_outfeeder.py + tests 5/5; spec file reports/BO-OUTFEEDER-SPEC.md (skopt GP defaults verified in source)
Task 6: Ruling: dashboard backend on 8034 temporarily JAX_PLATFORMS=cpu — first retune OOMed on the GPU (2.76 GiB) while 8 BO workers hold ~13 GB; restore GPU backend after the BO — cost: slower live retunes for ~2 h
Task 6: Ruling: "Best possible" was the paper's shared-pair optimum (per-zone looked -36 % "above" it); reference now = same structure on the true plant (out-feeder HGS with saved ends / 6-D search) — test_best_possible_is_the_same_structure_on_the_true_plant RED->GREEN; two-stage tests 8/8
Task 6: live check (CPU backend): first case saves six gains (UW 6.48/12.0, RW 20.27/11.4), out-feeder 3.89/30; new drift reuses them (6-D evals 0), out-feeder 3.93->4.255, S 0.843->0.521, ratio to floor 1.0002. Default window height 580->720 so the saved-gains panel is not clipped.
Final review: opus fresh reviewer — Ready with fixes; 0 critical, 5 important
Final: fixed I2 store temp-name collision + corrupt record — test_concurrent_saves_of_one_plant_do_not_collide, test_an_unreadable_record_reads_as_nothing_saved RED->GREEN
Final: fixed I1 forget-then-same-click claimed saved over empty store — test_cache_hit_recommits_after_forget RED->GREEN
Final: fixed I4 (part) six gains saved even when step 7 restored — _commit_per_zone saves only accepted cases; test_restored_case_does_not_save_the_six RED->GREEN; frontend note shows "NOT saved: <reason>"
Final: Ruling: I4 (part) "gains today"/restore target stay the paper's shared commissioning pair even after the six are saved — plan said restore unchanged; ASK USER — cost if wrong: dashboard "doing nothing" differs from the BO campaign's commissioned-6 arm
Final: fixed I5 dashboard cost doc only described Eq. (12) — T1 row added in App.jsx (formula from retuning_tiered_eval.py), Eq. (12) labelled T0; verified by build (no frontend test framework)
Final: fixed I3 slow route tests broken by new defaults — pinned to shared-2D/T0 (PAPER); tests/conftest.py session-wide store isolation; RED (3 failed, -m slow) -> rerun pending
Final: minor (deferred): full-6D cache hit of a pre-two-stage payload lacks per_zone_gains (latent; .get used now in the hit path)
Final: minor (deferred): double-stored payload keeps reused False / S_twin is the source case's twin cost (label S_twin_at_source)
Final: minor (deferred): FORGET button — no confirm in ALL mode, no try/catch, note never cleared, panel not refreshed; saved six only visible after OPTIMISE (zoneGainsFor wired, unused)
Final: minor (deferred): "Best possible (same gains fixed)" wording wrong for full-6D
Final: minor (deferred): BO spec calls HGS floor "best any out-feeder method can do" — BO beats it in some cells; handled in the BO report (negative gaps reported, wording fixed)
Final: minor (deferred): star duplicates centre when T_I sits on the 30 s ceiling (spends one eval) — conforms to spec; noted in report
Final: minor (deferred): run_bo_outfeeder cell temp name fixed / write outside try; HGS+BO record keeps BO's kp/ti when HGS wins
Final: minor (deferred): Task 5 uses POST /zone-gains + /zone-gains/delete (plan said GET/DELETE); frontend never sends cost_tier
Final: minor (deferred): launcher set -e hides commissioning failure reason (stderr log only)
Final: suites after fix pass — normal 430 passed / 66 skipped / 0 failed; slow 11 passed / 1 failed = the documented pre-existing golden drift (payload/validate.excitation/__digest, PROGRESS §6), 285/286 unchanged
Task 1: Ruling: pass-1 done 02:02 (P189 10/10); 4 reverse-order helpers on P049 alongside the 2 pass-2 workers — duplicates at most where they meet, deterministic results — cost: one wasted cell
Task 6: Ruling: verification runs saved six gains for P01/P02 from drifts I chose; forgotten via the API so the user's first case commissions each plant — cost: one 6-D search on the user's next click
Final: Ruling: workspace (ledger) kept, not deleted — no git, so it is the only record of rulings — cost: one small directory
Task 7: complete (BO-OUTFEEDER-RESULTS.pdf 4 pp via report workflow + 3 critics + condense pass; spec PDF 2 pp; figure symlog box; aggregation tests 6/6)
Task 8: Ruling: precompute NOT relaunched — under the new defaults it would commission each plant from its first cell; left as user decision (b) — cost: clicks compute live (~30-60 s) until relaunched
