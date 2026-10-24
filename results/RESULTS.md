# Results summary (auto-generated)

Run settings: repeats=10, gaps per length=20, seed=2026,
quick=False. Primary sensor 6012002000326, secondary sensor 6012002000869.

## Data
- CSV rows: 179447; temperature readings after deduplication:
  29038 (primary), 13800 (secondary).
- Primary: 32707 slots, 3669 missing
  (11.2%),
  2025-09-07 20:09:30+10:00 to 2026-04-22 23:12:41+10:00.
- Secondary: 15191 slots, 1391 missing.

## Settings chosen on the validation period
k_daily=2, k_weekly=2, window_days=7.0,
hybrid switch_len=6 slots. **Selected method: hybrid**
(fastest method within 2% of the lowest mean validation MAE).

| method | mean_MAE | seconds |
|---|---|---|
| boosted | 0.832 | 0.838 |
| cubic_spline | 2.451 | 0.003 |
| ffill | 1.569 | 0.001 |
| harmonic | 0.740 | 0.793 |
| hybrid | 0.740 | 0.022 |
| linear | 1.112 | 0.001 |
| pchip | 1.098 | 0.002 |

## Held-out test MAE (°C), primary sensor
| gap_hours | ffill | linear | cubic_spline | pchip | harmonic | hybrid | boosted |
|---|---|---|---|---|---|---|---|
| 0.167 | 0.097 | 0.063 | 0.084 | 0.066 | 0.064 | 0.063 | 0.090 |
| 0.500 | 0.159 | 0.083 | 0.110 | 0.085 | 0.084 | 0.083 | 0.091 |
| 1.000 | 0.269 | 0.108 | 0.166 | 0.117 | 0.109 | 0.108 | 0.113 |
| 3.000 | 0.669 | 0.202 | 0.384 | 0.219 | 0.189 | 0.189 | 0.205 |
| 6.000 | 1.070 | 0.495 | 0.784 | 0.469 | 0.415 | 0.415 | 0.434 |
| 12.000 | 1.664 | 0.904 | 1.331 | 0.859 | 0.643 | 0.643 | 0.684 |
| 24.000 | 2.051 | 1.720 | 2.880 | 1.724 | 0.951 | 0.951 | 0.942 |
| 48.000 | 2.470 | 1.825 | 5.778 | 1.849 | 1.369 | 1.369 | 1.327 |
| 72.000 | 2.474 | 1.893 | 8.713 | 1.960 | 1.812 | 1.812 | 1.759 |

Best method per gap length:

| gap_hours | method | MAE |
|---|---|---|
| 0.167 | linear | 0.063 |
| 0.500 | linear | 0.083 |
| 1.000 | linear | 0.108 |
| 3.000 | harmonic | 0.189 |
| 6.000 | harmonic | 0.415 |
| 12.000 | harmonic | 0.643 |
| 24.000 | boosted | 0.942 |
| 48.000 | boosted | 1.327 |
| 72.000 | boosted | 1.759 |

Selected method meets the 0.5 °C acceptance MAE up to 6 h gaps.

## Generalisation MAE (°C), secondary sensor
| gap_hours | ffill | linear | cubic_spline | pchip | harmonic | hybrid | boosted |
|---|---|---|---|---|---|---|---|
| 0.167 | 0.073 | 0.046 | 0.065 | 0.048 | 0.047 | 0.046 | 0.079 |
| 0.500 | 0.120 | 0.079 | 0.109 | 0.080 | 0.081 | 0.079 | 0.102 |
| 1.000 | 0.227 | 0.123 | 0.147 | 0.111 | 0.124 | 0.123 | 0.143 |
| 3.000 | 0.516 | 0.206 | 0.336 | 0.205 | 0.217 | 0.217 | 0.234 |
| 6.000 | 0.686 | 0.429 | 0.551 | 0.396 | 0.349 | 0.349 | 0.370 |
| 12.000 | 1.031 | 0.729 | 1.339 | 0.732 | 0.448 | 0.448 | 0.519 |
| 24.000 | 1.110 | 1.078 | 2.562 | 1.073 | 0.576 | 0.576 | 0.697 |
| 48.000 | 1.123 | 1.067 | 3.984 | 1.067 | 0.635 | 0.635 | 0.821 |
| 72.000 | 1.189 | 1.165 | 6.239 | 1.166 | 0.699 | 0.699 | 1.109 |

## Performance (filling all real gaps of the primary sensor)
| method | seconds | peak_MB | flops | MFLOP_per_s | gaps_fitted |
|---|---|---|---|---|---|
| ffill | 0.0004484 | 1.116 | nan | nan | nan |
| linear | 0.0004513 | 1.604 | 1.834e+04 | 40.65 | 0 |
| cubic_spline | 0.002626 | 5.236 | 2.69e+05 | 102.4 | 0 |
| pchip | 0.002036 | 4.074 | 6.174e+05 | 303.3 | 0 |
| harmonic | 0.838 | 6.617 | 9.814e+08 | 1171 | 2143 |
| hybrid | 0.0239 | 6.617 | 1.366e+07 | 571.8 | 15 |

Loop vs vectorised linear interpolation: 0.143 s vs 0.02324 s
(speed-up 6x, identical output: True).

Boosted model: fit 31.4 s on 78641 rows;
fill 0.84 s.

Scaling exponents (slope of log time vs log n):

| method | exponent | n_range |
|---|---|---|
| linear | 0.969 | 64000-1024000 |
| cubic_spline | 0.969 | 64000-1024000 |
| pchip | 0.969 | 64000-1024000 |
| hybrid | 0.994 | 64000-1024000 |
| harmonic | 1.031 | 16000-256000 |

Least-squares solver comparison:

| trend_centred | solver | cond_A | cond_AtA | seconds | max_coef_diff_vs_svd | rows_m | params_p |
|---|---|---|---|---|---|---|---|
| False | lstsq | 1.59e+04 | 2.51e+08 | 0.000164 | 0 | 1336 | 10 |
| False | normal | 1.59e+04 | 2.51e+08 | 3.28e-05 | 2.44e-11 | 1336 | 10 |
| True | lstsq | 4.94 | 24.4 | 0.000164 | 0 | 1336 | 10 |
| True | normal | 4.94 | 24.4 | 3.28e-05 | 1.58e-13 | 1336 | 10 |

## Reconstruction of real gaps
| sensor | filled_slots | longest_gap_slots |
|---|---|---|
| 6012002000326 | 3669 | 646 |
| 6012002000869 | 1391 | 2 |
