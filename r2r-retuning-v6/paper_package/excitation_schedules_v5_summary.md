# Excitation schedules (paper v5) — measured from code

Common conventions: plant integrates at dt = 1 ms, controller at T_s = dt (ZOH); every tension step is +20% of that channel's setpoint; episodes last 5 s; the full record including the settle window enters the estimator (nothing is discarded).

| Type | Campaign | Records | Duration [s] | Edges (t: channel dir) |
|------|----------|---------|--------------|------------------------|
| ET1 | A_tension_factorial | 1 | 7 | 2: span1 ^ |
| ET3 | A_tension_factorial | 1 | 17 | 2: span1 ^; 7: span2 ^; 12: span3 ^ |
| ET6 | A_tension_factorial | 1 | 32 | 2: span1 ^; 7: span2 ^; 12: span3 ^; 17: span1 v; 22: span2 v; 27: span3 v |
| E_Toggle | A_tension_factorial | 1 | 17 | 2: span1 ^; 7: span1 v; 7: span2 ^; 12: span1 ^; 12: span2 v; 12: span3 ^ |
| ET3M | A_tension_factorial | 3 | 3 x 17 = 51 total | 2: span1 ^; 7: span2 ^; 12: span3 ^ |
| EV1 | A_tension_factorial | 1 | 12 | 2: v_line ^ |
| ET1 | B_dual_channel | 1 | 30 | 2: span1 ^ |
| E_Toggle | C_retuning_field_matched | 1 | 16 | 1: span1 ^; 6: span1 v; 6: span2 ^; 11: span1 ^; 11: span2 v; 11: span3 ^ |

Seed conventions (verified on the campaign CSVs):
- dual/composite: `seed_v = seed_T + 100` (R13 `COMPOSITE_SEED_V_OFFSET`); velocity-only rows have `seed_T = NaN`.
- ET3M record i: `seed_T = base_seed + 17*i`.

Notes:
- Group B (dual) differs from group A only in ET1 (30 s vs 7 s record — reproduction gate); the other five types are identical.
- Group C is the main-text §4 identification protocol (E_Toggle, settle 1 s, 16 s record, edges at 1/6/11 s).
- EV1 excites the line speed only (channel `v_line`); its tension references stay at the setpoints for the whole record.
