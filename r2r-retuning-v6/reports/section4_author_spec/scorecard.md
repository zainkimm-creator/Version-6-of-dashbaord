# Scorecard — section4_author_spec

cells ok 120, failed 0; devices cuda:0; config 9202a65abc407933

Verdicts: REPRODUCED 80, CLOSE 9, DIFFERS 6, no target 0, not computed 0

| Section | Item | Paper | Authors' reply | Ours | Verdict | Note |
|---|---|---|---|---|---|---|
| Table 3 / S9 | CS-BO(30) n | 180 | — | 180 | REPRODUCED |  |
| Table 3 / S9 | CS-BO(30) median S | 0.407 | — | 0.4074 | REPRODUCED |  |
| Table 3 / S9 | CS-BO(30) mean S | 0.687 | — | 0.687 | REPRODUCED |  |
| Table 3 / S9 | CS-BO(30) P5 | 0.109 | — | 0.1088 | REPRODUCED |  |
| Table 3 / S9 | CS-BO(30) P95 | 2.634 | — | 2.633 | REPRODUCED |  |
| Table 3 / S9 | WS-BO(30) n | 180 | — | 180 | REPRODUCED |  |
| Table 3 / S9 | WS-BO(30) median S | 0.407 | — | 0.4079 | REPRODUCED |  |
| Table 3 / S9 | WS-BO(30) mean S | 0.689 | — | 0.6878 | REPRODUCED |  |
| Table 3 / S9 | WS-BO(30) P5 | 0.11 | — | 0.1097 | REPRODUCED |  |
| Table 3 / S9 | WS-BO(30) P95 | 2.642 | — | 2.63 | REPRODUCED |  |
| Table 3 / S9 | HGS-only n | 60 | — | 60 | REPRODUCED |  |
| Table 3 / S9 | HGS-only median S | 0.357 | — | 0.3477 | CLOSE |  |
| Table 3 / S9 | HGS-only mean S | 0.695 | — | 0.6861 | REPRODUCED |  |
| Table 3 / S9 | HGS-only P5 | 0.109 | — | 0.1087 | REPRODUCED |  |
| Table 3 / S9 | HGS-only P95 | 2.688 | — | 2.658 | REPRODUCED |  |
| Table 3 / S9 | HGS+BO(5) n | 60 | — | 60 | REPRODUCED |  |
| Table 3 / S9 | HGS+BO(5) median S | 0.357 | — | 0.3477 | CLOSE |  |
| Table 3 / S9 | HGS+BO(5) mean S | 0.692 | — | 0.6861 | REPRODUCED |  |
| Table 3 / S9 | HGS+BO(5) P5 | 0.109 | — | 0.1087 | REPRODUCED |  |
| Table 3 / S9 | HGS+BO(5) P95 | 2.688 | — | 2.658 | REPRODUCED |  |
| Table 3 / S9 | HGS+BO(10) n | 60 | — | 60 | REPRODUCED |  |
| Table 3 / S9 | HGS+BO(10) median S | 0.357 | — | 0.3477 | CLOSE |  |
| Table 3 / S9 | HGS+BO(10) mean S | 0.69 | — | 0.6838 | REPRODUCED |  |
| Table 3 / S9 | HGS+BO(10) P5 | 0.109 | — | 0.1087 | REPRODUCED |  |
| Table 3 / S9 | HGS+BO(10) P95 | 2.688 | — | 2.658 | REPRODUCED |  |
| Pooled medians | HGS-only rank 30 (of 60) | — | 0.3043 | 0.284 | DIFFERS |  |
| Pooled medians | HGS-only rank 31 (of 60) | — | 0.41 | 0.4114 | REPRODUCED |  |
| Win rate (5 slow plants) | HGS-only < median CS-BO(30), cells | 29 | 29 | 29 | REPRODUCED | 29/50 = 58 % (paper 58 %) |
| Win rate (5 slow plants) | HGS+BO(5) < median CS-BO(30), cells | — | 29 | 29 | REPRODUCED | 29/50 |
| Win rate (5 slow plants) | HGS+BO(10) < median CS-BO(30), cells | — | 29 | 29 | REPRODUCED | 29/50 |
| Win rate per plant | P001 wins /10 | — | 5 | 6 | REPRODUCED |  |
| Win rate per plant | P049 wins /10 | — | 10 | 10 | REPRODUCED |  |
| Win rate per plant | P053 wins /10 | — | 6 | 7 | REPRODUCED |  |
| Win rate per plant | P158 wins /10 | — | 5 | 3 | CLOSE |  |
| Win rate per plant | P186 wins /10 | — | 3 | 3 | REPRODUCED |  |
| Win rate vs baseline budget | CS-BO at 10 evals: HGS-only win % | — | 100 | 100 | REPRODUCED |  |
| Win rate vs baseline budget | CS-BO at 20 evals: HGS-only win % | — | 88 | 84 | REPRODUCED |  |
| Win rate vs baseline budget | CS-BO at 25 evals: HGS-only win % | — | 68 | 72 | REPRODUCED |  |
| Win rate vs baseline budget | CS-BO at 30 evals: HGS-only win % | — | 58 | 58 | REPRODUCED |  |
| Parity diagnostics | paired rel. diff median % | — | -0.04 | -0.05872 | REPRODUCED |  |
| Parity diagnostics | paired rel. diff P25 % | — | -0.24 | -0.3466 | REPRODUCED |  |
| Parity diagnostics | paired rel. diff P75 % | — | 0.23 | 0.2803 | REPRODUCED |  |
| Parity diagnostics | paired median HGS-only | — | 0.2137 | 0.2138 | REPRODUCED |  |
| Parity diagnostics | paired median CS-BO(30) | — | 0.214 | 0.2146 | REPRODUCED |  |
| Parity diagnostics | sign test p (two-sided) | — | 0.32 | 0.3222 | REPRODUCED |  |
| Parity diagnostics | 1 % band: HGS better | — | 7 | 7 | REPRODUCED |  |
| Parity diagnostics | 1 % band: tie | — | 36 | 39 | REPRODUCED |  |
| Parity diagnostics | 1 % band: CS-BO better | — | 7 | 4 | REPRODUCED |  |
| Parity diagnostics | within 1 % of CS-BO, % cells | — | 86 | 78 | CLOSE |  |
| Parity diagnostics | within 5 % of CS-BO, % cells | — | 98 | 92 | CLOSE |  |
| Budget-matched (paper 4.2) | HGS+BO(5) < CS-BO at 5 evals, cells | 60 | 60 | 60 | REPRODUCED | paper: 'beats cold-start BO in every paired run' |
| Budget-matched (paper 4.2) | HGS+BO(5) < CS-BO at 30 evals, cells (all 60) | — | 29 | 31 | REPRODUCED |  |
| CS-BO(30) vs transfer | per-cell CS-BO seed-median / HGS-only, median | — | 0.9999 | 1 | REPRODUCED |  |
| CS-BO(30) vs transfer | cells where CS-BO better (of 60) | — | 31 | 29 | REPRODUCED |  |
| CS-BO(30) vs transfer | real evaluations per cell | — | 196 | 196 | REPRODUCED |  |
| CS-BO(30) vs transfer | cells with CS-BO within 1 % of best-of-196 | — | 52 | 52 | REPRODUCED |  |
| CS-BO(30) vs transfer | anchored CS-BO runs improving between evals 25 and 30, % | — | 50 | 53.33 | REPRODUCED |  |
| CS-BO(30) vs transfer | median gain of those runs, % | — | 0.66 | 0.689 | REPRODUCED |  |
| Few-shot BO | HGS+BO(5) improved on transfer, cells | — | 3 | 1 | CLOSE | P158/D02 +0.01% |
| Few-shot BO | HGS+BO(10) improved on transfer, cells | — | 5 | 14 | DIFFERS | P001/D08 +0.97%; P001/D10 +0.05%; P049/D03 +0.05%; P053/D05 +0.08%; P158/D01 +0.05%; P158/D02 +0.02%; P158/D07 +0.06%; P158/D10 +0.04%; P186/D02 +0.24%; P186/D03 +0.01%; P186/D05 +0.03%; P186/D08 +0.08%; P186/D09 +0.07%; P189/D07 +4.06% |
| Few-shot BO | HGS+BO(10) star duplicates centre (eval 5 = eval 1), cells | — | 59 | 47 | DIFFERS |  |
| Few-shot BO | HGS+BO(10) first EI step at K_p* = 1 bound, cells | — | 54 | 52 | REPRODUCED |  |
| Few-shot BO | arm-cells where an EI step delivered the final best (of 120) | — | 2 | 2 | REPRODUCED | HGS+BO(5) has no EI step |
| WS-BO seeding (Fig. 7a) | eval 1 median | — | 17.14 | 17.14 | REPRODUCED |  |
| WS-BO seeding (Fig. 7a) | running best, evals 2-5 median | 10.7 | 10.71 | 10.71 | REPRODUCED |  |
| WS-BO seeding (Fig. 7a) | running best at eval 6 median | — | 2.75 | 2.752 | REPRODUCED |  |
| WS-BO seeding (Fig. 7a) | K_p* -30 % point improves running best, runs | — | 180 | 180 | REPRODUCED |  |
| WS-BO seeding (Fig. 7a) | other star points improve running best, runs | — | 0 | 0 | REPRODUCED |  |
| WS-BO seeding (Fig. 7a) | eval 5 repeats eval 1 (T_I clipped), runs | — | 120 | 120 | REPRODUCED |  |
| WS-BO seeding (Fig. 7a) | seed 0: first random draw improves, cells | — | 60 | 60 | REPRODUCED |  |
| WS-BO seeding (Fig. 7a) | seed 2: first random draw improves, cells | — | 41 | 41 | REPRODUCED |  |
| WS-BO seeding (Fig. 7a) | seed 1: first random draw improves, cells | — | 0 | 0 | REPRODUCED |  |
| Fig. 7a | CS-BO median running best at 5 evals | 0.878 | — | 0.8776 | REPRODUCED |  |
| Fig. 7a | CS-BO median running best at 30 evals | 0.407 | — | 0.4074 | REPRODUCED |  |
| Fig. 7a | CS-BO curve: max rel. deviation over 30 evals | 0 | — | 0.002907 | REPRODUCED | authors' figure data (fig7a_convergence_reference.csv) |
| Fig. 7a | WS-BO curve: max rel. deviation over 30 evals | 0 | — | 0.002679 | REPRODUCED |  |
| P189 exception | HGS-only wins on P189, cells | 0 | — | 2 | DIFFERS |  |
| P189 exception | P189 HGS-only median S | 2.53 | — | 2.507 | REPRODUCED |  |
| P189 exception | P189 CS-BO(30) median S | 2.49 | — | 2.493 | REPRODUCED |  |
| P189 exception | P189 twin parameter error (median %) | 293 | — | 282.1 | REPRODUCED |  |
| Logging-only protocol | HGS-only win rate % | 5 | — | 8.333 | REPRODUCED | 5/60 |
| Logging-only protocol | HGS-only median S | 0.557 | — | 0.5555 | REPRODUCED | paper prints this as a paired-scope median |
| Logging-only protocol | CS-BO median S (per-cell seed medians) | 0.419 | — | 0.4196 | REPRODUCED |  |
| Sim-to-real gap | logging-only median gap %, (S_twin - S_plant) / S_plant | -16.2 | — | 20.09 | DIFFERS | first reading tried; opposite sign to the paper |
| Sim-to-real gap | logging-only median gap %, (S_plant - S_twin) / S_twin | -16.2 | — | -16.73 | REPRODUCED | definition INFERRED: the paper prints only the medians; this reading matches both of them |
| Sim-to-real gap | field-matched median gap %, (S_twin - S_plant) / S_plant | -1.16 | — | 0.8959 | CLOSE | first reading tried |
| Sim-to-real gap | field-matched median gap %, (S_plant - S_twin) / S_twin | -1.16 | — | -0.8879 | REPRODUCED | definition INFERRED (see logging-only line) |
| Evaluator facts | twin optima with T_I on the 30 s bound (of 60) | — | 59 | 47 | DIFFERS |  |
| Evaluator facts | delivered runs with T_I at 30 s (of 540) | — | 320 | 289 | CLOSE |  |
| Evaluator facts | plants whose nominal auto_Ti exceeds 30 s (of 6) | — | 4 | 4 | REPRODUCED | P001 18.5 s, P049 36.8 s, P053 37.0 s, P158 36.3 s, P186 49.8 s, P189 17.7 s |
| Evaluator facts | delivered gain pairs with OS <= 8.4 %, % | — | 98 | 98.52 | REPRODUCED |  |
| Evaluator facts | median OS at delivered gains, % | — | 2.36 | 2.337 | REPRODUCED |  |
| Evaluator facts | overshoot term share of median cost, % | — | 7 | 6.7 | REPRODUCED |  |
| Evaluator facts | integrator peak at delivered HGS gains, max N s | — | 20 | 21.54 | REPRODUCED | a +-200 N s clamp is inert if this stays far below 200 |
| Evaluator facts | non-finite (diverging) evaluations, whole campaign | — | 0 | 0 | REPRODUCED |  |
