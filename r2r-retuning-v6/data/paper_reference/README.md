# Paper reference data

Every file here is **comparison-only**. Paper values never feed a dashboard
calculation: the dashboard recomputes each result from the model and reports the
difference. `provenance.paper_used_for_calculation` is asserted `false` on every
live result.

All values track **paper1_isa_v5** (main text + supplement, 2026-08-03).

## Why v4.1 numbers are not comparable

v5 was not a wording pass. The estimator itself changed, every campaign was
re-run under it, and a measurement condition that did not previously exist
(velocity-channel noise) was added. Concretely:

- Identification now minimizes an **operating-point-weighted** one-step
  prediction error over the **six-channel** logged state `z` (three span
  tensions and three roller angular velocities), not the tension-only output.
- `W = diag(1/T_ref,1..3, 1/omega_ss,UW, 1/omega_ss,Nip, 1/omega_ss,RW)`.
- The Fisher information is `F = sum_k J_k^T W^2 J_k`, and `kappa(F)` is
  reported as a **ranking only**.
- `T_s = dt = 1 ms` in all campaigns, not 10 ms.
- Results are scoped to one of three conditions: noise-free, tension-only, or
  dual-channel. The v4.1 "SN" column was tension-only at LPF 100 Hz; the v5 "SN"
  column in the excitation table is dual-channel at LPF 50 Hz. **Different
  condition, not a corrected value.**

Three different measurement conditions coexist in the paper — the excitation
table is dual-channel, while the gain sweep and the drift campaign remain
tension-only at 100 Hz — so every stored number carries its condition tag.

## Files

| File | Contents | Paper source |
|---|---|---|
| `excitation_reference.json` | Six excitation types, NF and SN medians, off-scale 75th percentiles, conditioning ranking | Table 1, Fig. 3, Table S6 |
| `logging_rate_v5_reference.json` | Logging-period curves under all three conditions, interior optimum, slow/fast decomposition | Fig. 2, Table S7, Fig. S6 |
| `logging_power_law_reference.json` | The `tau_min`-normalized noise-free power law | Section S3.1, Fig. S2 |
| `drift_reference.json` | EA / f / J drift levels, per-roller diagnostic, reel-radius sensitivity | Section 3.3, Fig. 4 |
| `noise_lpf_reference.json` | Feasibility gate, main-effect spreads, transition table, LPF x Tlog heatmap, cross-channel ratios | Section 3.4, Section S7, Section S6 |
| `closed_loop_damping_reference.json` | `K_p*` gain sweep by damping group, Simpson's-paradox note | Section 3.5, Fig. 6 |
| `retuning_reference.json` | Digital-twin retuning by method and protocol, tail risk, P189 exception | Tables 2-3, Fig. 7, Table S8 |
| `experiment_ledger_v5.json` | All 16 campaigns with their grid decompositions | Table S10 |
| `figures_v5.json` | The 17 v5 figures and the v4.1 -> v5 renumbering map | Section 7 |
| `paper1_isa_supplement_parameters.json` | Plant/supplement parameters | Supplement |
| `logging_adequacy_fig02_reference.csv` | Digitized logging-adequacy points | (legacy, v4.1) |
| `logging_rate_v41_reference.json` | Superseded by `logging_rate_v5_reference.json` | (legacy, v4.1) |

## Values the paper deliberately does not publish

These appear as `null`, and that is correct. Do not synthesize them.

- **Per-leg EA drift medians.** The paper reports the EA family as a band across
  its five legs (NF 24.4-25.2 %, SN 28.6-31.4 %) because the median is flat
  within roughly a percentage point and the counter-effect is non-monotone.
- **The two damping-group values at `K_p* = 100`.** Only the endpoints at 50 and
  200 are printed, together with the pooled value at all three gains.
- **The SN pair for the friction drift legs.** Only the noise-free pair is given.
- **A grand run total.** The ledger prints each campaign's grid decomposition
  instead, by explicit authorial choice.
- **Absolute `kappa(F)` values.** Rankings only.
- **The overshoot cost of an off-default gain.** The gain sweep logs no transient
  metric.
