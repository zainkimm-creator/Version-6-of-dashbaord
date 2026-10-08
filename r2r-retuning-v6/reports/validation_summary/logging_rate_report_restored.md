# Logging Adequacy Report

- Tlog values: 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0 ms
- tmin / tau_min: 50.0 ms
- Measurement conditions: noise_free, tension_only, dual_channel
- Dual-channel optimum Tlog: 5.0 ms
- Dual-channel optimum MARE_theta: 28.540331993948307%
- Dual-channel optimum Tlog/tmin: 0.22806995748432013
- Dual-channel optimum tau_min/Tlog: 4.38461957475811
- Tension-only best Tlog: 1.0 ms (no interior optimum; the finest setting always wins)
- CSV summary: `reports/validation_summary/logging_rate_summary_restored.csv`

## Fig. 2(a) | Three measurement conditions

![Logging period under three measurement conditions](../figures/logging_rate_vs_mare_restored.svg)

## Fig. 2(b) | Dual-channel speed decomposition

![Dual-channel decomposed by line speed](../figures/logging_rate_speed_decomposition.svg)

The pooled valley represents neither sub-group and is never reported without this split.

## Fig. S2 | Noise-free power law

![Noise-free tau_min-normalized power law](../figures/logging_rate_power_law_restored.svg)

The sensor-noise branch is deliberately not overlaid on the power law: under tension-only sensor noise the error does not collapse onto it (fitted exponent +0.41, R^2 ~ 0.11), so an SN fit would contradict its own trend.

## Power-Law Fits

| Fit | Case | Source | Branch | a | alpha | R^2 | Points |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Median NF | noise_free | dashboard_run | per_plant_n60_excl_1ms | 23.5419 | 0.966091 | 0.790721 | 60.0 |
| Paper NF (MARE) | noise_free | paper_reference | all_runs | 23.5 | 0.97 | 0.79 | 6.0 |

Model form: `MARE_theta_percent = a * (Tlog/tau_min)^alpha`.
The power law is fitted on the noise-free branch only, per Fig. S2.
