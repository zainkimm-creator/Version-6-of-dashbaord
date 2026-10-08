# Paper v5 — simulation inputs for the validation dashboard

Response to your two data requests of 2026-08-12 ("Data Needed").
Everything here is extracted deterministically from the frozen canonical
sources in the repository; the producer scripts are committed alongside
the paper so each file can be regenerated bit-identically.

## 1. `ten_plant_parameters.csv` — per-plant physical parameters

One row per plant of the fixed ten-plant subset
(P001, P049, P053, P060, P139, P158, P163, P177, P186, P189).

| Column group | Meaning |
|---|---|
| `EA_N` | web axial stiffness [N] |
| `L1_m, L2_m, L3_m` | span lengths [m], numbered UW→Feeder / Feeder→Nip / Nip→RW |
| `v0_mps` | nominal line speed (in-feeder BC) [m/s]; `v_max_mps = v0 / 0.30` |
| `T_ref_N` | set-point tension [N] — **uniform across the three channels**; `T_max_N = T_ref / 0.30` |
| `R_*_m`, `J_*_kgm2`, `f_*_Nms_per_rad` | roller radius / inertia / viscous friction, per roller (UW, Nip, RW) |
| `omega_n_*_rad_s` | per-roller natural frequency `sqrt(EA·R²/(J·L))` |
| `omega_ss_*`, `u_ss_*` | steady-state angular velocity / torque at the operating point |
| `K_vel_*` | inner-loop velocity gains `1.4 · J · omega_n` (the campaign default) |
| `T_I_s` | integral time from the magnitude heuristic (`auto_Ti`, settle 5 s, floor 0.1 s) |
| `regime_paper` | damping label as printed in the paper (O-UD / H-Osc / H-Damp) |

Self-checks the extractor runs before writing (all PASS): EA range
3.2–1,242 kN, J 0.0975–937.7 kg m², f 0.126–10.0, R 0.10–0.75 m match
Supplementary Table S5; ζ_CL,min band 0.14–0.53; regime split 6 O-UD /
2 H-Osc / 2 H-Damp with H-Damp = {P158, P186}.

**Pitfalls to avoid when cross-checking against the repo:**
- The pool JSON carries two *legacy* fields the v5 campaigns do **not**
  use: `noise_sigma` and `step_size`. Noise is set per-run as a fraction
  of `T_max` (sweep axis 0.02–0.5 %), and every tension step is 20 % of
  the channel setpoint. Reading `step_size` from the JSON would halve
  your step.
- Ignore any file under `research/R12_plant_overhaul/data/old/` — it is a
  stale pre-v3 pool that reuses the same plant IDs with entirely
  different values and a different ten-plant subset.
- The value `T_ref,0 = 30 N` printed in the Fig. 9 caption belongs to
  plant **P007**, an illustrative pool draw that is *not* in the
  ten-plant set — do not use it as a set-point anywhere.

## 2. `excitation_schedules_v5.csv` — exact excitation schedules

Edge-level long format: one row per reference change, measured by
scanning the excitation callables on the campaign integration grid
(dt = 1 ms). `excitation_schedules_v5_summary.md` gives the one-row-
per-type human-readable version.

- `campaign_group` — `A_tension_factorial` (the 13,020-run sweep),
  `B_dual_channel` (the dual grids; only ET1 differs, 30 s instead of
  7 s, kept for bit-exact reproduction of the velocity-noise reference
  cells), `C_retuning_field_matched` (main-text §4: E_Toggle with a 1 s
  settle → 16 s record, edges at 1/6/11 s).
- `edge_channel` — `span1..span3` for tension steps (spans numbered
  from the unwinder), `v_line` for the EV1 line-speed step.
- Amplitudes: every tension step is +20 % of that channel's setpoint;
  EV1 steps the line speed v0 → 1.2 v0 and leaves all tension
  references untouched (the PEM record splits at the step time).
- The full record, including the settle window, enters the estimator —
  nothing is discarded.
- Seed conventions: dual/composite runs use `seed_v = seed_T + 100`;
  velocity-only rows carry `seed_T = NaN`; ET3M record *i* uses
  `seed_T = base + 17·i`.

These schedules are also now printed in the paper as Supplementary
Table S1 (Section S1.1), added in the same revision that answers your
three "Difference found" questions.

## Provenance

| File | Producer (in repo) | Source of truth |
|---|---|---|
| `ten_plant_parameters.csv` | `papers/paper1_sysid_adaptive/isa_v2/fig_src/extract_ten_plant_params.py` | `research/R12_plant_overhaul/plant_pool_v3.json` (md5 `008355c7a0b059ff6e114a3870c7c071`) + `run_full_sweep.SELECTED` |
| `excitation_schedules_v5.csv` / `_summary.md` | `research/R12_plant_overhaul/dump_excitation_schedules.py` | `excitations3.py` factories scanned at dt = 1 ms |
