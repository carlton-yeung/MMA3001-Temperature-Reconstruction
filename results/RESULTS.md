# Results summary (auto-generated)

Run: repeats short/long = 10/20, gaps per length = 20,
seed = 2026, quick = False.

## Data
- CSV rows: 179447.
- Primary 6012002000869 (B60 G.38 Compactus_upper): 13800 readings,
  15191 slots, 1391 missing (9.2%), 2024-02-29 17:04:56+11:00 to 2024-06-14 03:45:44+10:00.
- Secondary 6012002000326 (location unknown): 29038 readings,
  32707 slots, 3669 missing (11.2%).

## Decisions made on the development period (first 60% of primary)
- Harmonic settings: K = 6 daily, J = 0 weekly harmonics,
  window = 3.0 days.
- Hybrid switch: linear up to 6 slots (1 h), harmonic above.
- Best single method: harmonic. Hybrid adopted on accuracy (> 0.08 °C gain
  somewhere, no loss > 0.08 °C anywhere): **False**.
- Final method by the selection rules: **harmonic**. Hybrid accuracy within
  2% of harmonic at every length: True.
  Deployed implementation: **hybrid**.

| method | mean_dev_MAE |
|---|---|
| cubic_spline | 0.448 |
| ffill | 0.475 |
| harmonic | 0.204 |
| hybrid | 0.204 |
| linear | 0.278 |
| pchip | 0.268 |
| yesterday | 0.479 |

## Held-out test MAE (°C), primary sensor
Up to 2 h: fully observed gaps; above 2 h: partially observed gaps.

| gap_hours | ffill | linear | yesterday | cubic_spline | pchip | harmonic | hybrid |
|---|---|---|---|---|---|---|---|
| 0.167 | 0.059 | 0.056 | 0.416 | 0.075 | 0.057 | 0.058 | 0.056 |
| 0.500 | 0.154 | 0.070 | 0.363 | 0.095 | 0.070 | 0.075 | 0.070 |
| 1.000 | 0.212 | 0.104 | 0.394 | 0.144 | 0.100 | 0.104 | 0.104 |
| 2.000 | 0.390 | 0.197 | 0.412 | 0.231 | 0.177 | 0.188 | 0.188 |
| 3.000 | 0.570 | 0.243 | 0.419 | 0.364 | 0.238 | 0.226 | 0.226 |
| 4.000 | 0.701 | 0.358 | 0.454 | 0.491 | 0.328 | 0.268 | 0.268 |
| 6.000 | 0.862 | 0.481 | 0.425 | 0.684 | 0.446 | 0.315 | 0.315 |
| 8.000 | 1.028 | 0.655 | 0.477 | 0.974 | 0.607 | 0.373 | 0.373 |
| 12.000 | 1.133 | 0.854 | 0.433 | 1.427 | 0.833 | 0.417 | 0.417 |

95th-percentile absolute error (°C), limit 1.0 °C:

| gap_hours | ffill | linear | yesterday | cubic_spline | pchip | harmonic | hybrid |
|---|---|---|---|---|---|---|---|
| 0.167 | 0.300 | 0.150 | 1.200 | 0.199 | 0.194 | 0.156 | 0.150 |
| 0.500 | 0.500 | 0.225 | 1.200 | 0.270 | 0.253 | 0.229 | 0.225 |
| 1.000 | 0.500 | 0.300 | 1.000 | 0.414 | 0.300 | 0.292 | 0.300 |
| 2.000 | 1.500 | 0.923 | 1.200 | 0.697 | 0.630 | 0.642 | 0.642 |
| 3.000 | 2.000 | 0.963 | 1.200 | 1.147 | 0.794 | 0.729 | 0.729 |
| 4.000 | 2.700 | 1.545 | 1.300 | 1.560 | 1.350 | 0.943 | 0.943 |
| 6.000 | 2.800 | 1.696 | 1.200 | 2.188 | 1.601 | 1.032 | 1.032 |
| 8.000 | 3.300 | 2.163 | 1.300 | 2.968 | 2.135 | 1.171 | 1.171 |
| 12.000 | 3.200 | 2.629 | 1.300 | 4.686 | 2.655 | 1.299 | 1.299 |

## Usable horizon (longest gap with upper 95% CI of MAE <= 0.5 °C)
| method | primary_0869_h | secondary_0326_h |
|---|---|---|
| ffill | 2 | 1 |
| linear | 4 | 4 |
| yesterday | 6 | 0 |
| cubic_spline | 3 | 3 |
| pchip | 6 | 4 |
| harmonic | 12 | 4 |
| hybrid | 12 | 4 |

## Agreement of the two gap designs (primary, test period)
| method | gap_hours | MAE_full | MAE_lo_full | MAE_hi_full | MAE_partial | MAE_lo_partial | MAE_hi_partial | CIs_overlap |
|---|---|---|---|---|---|---|---|---|
| ffill | 1.000 | 0.212 | 0.176 | 0.257 | 0.221 | 0.190 | 0.260 | True |
| linear | 1.000 | 0.104 | 0.090 | 0.119 | 0.113 | 0.101 | 0.126 | True |
| yesterday | 1.000 | 0.394 | 0.344 | 0.454 | 0.404 | 0.366 | 0.451 | True |
| cubic_spline | 1.000 | 0.144 | 0.129 | 0.160 | 0.141 | 0.130 | 0.152 | True |
| pchip | 1.000 | 0.100 | 0.089 | 0.112 | 0.103 | 0.094 | 0.112 | True |
| harmonic | 1.000 | 0.104 | 0.093 | 0.115 | 0.115 | 0.106 | 0.127 | True |
| hybrid | 1.000 | 0.104 | 0.090 | 0.119 | 0.113 | 0.101 | 0.126 | True |
| ffill | 2.000 | 0.390 | 0.332 | 0.458 | 0.382 | 0.340 | 0.428 | True |
| linear | 2.000 | 0.197 | 0.159 | 0.235 | 0.168 | 0.151 | 0.187 | True |
| yesterday | 2.000 | 0.412 | 0.364 | 0.467 | 0.374 | 0.344 | 0.406 | True |
| cubic_spline | 2.000 | 0.231 | 0.208 | 0.257 | 0.241 | 0.224 | 0.258 | True |
| pchip | 2.000 | 0.177 | 0.151 | 0.203 | 0.158 | 0.146 | 0.171 | True |
| harmonic | 2.000 | 0.188 | 0.165 | 0.214 | 0.164 | 0.150 | 0.177 | True |
| hybrid | 2.000 | 0.188 | 0.163 | 0.214 | 0.164 | 0.151 | 0.178 | True |

## Independent sensor and long-gap case study: MAE (°C), sensor 6012002000326
| gap_hours | ffill | linear | yesterday | cubic_spline | pchip | harmonic | hybrid |
|---|---|---|---|---|---|---|---|
| 0.167 | 0.089 | 0.065 | 2.345 | 0.088 | 0.069 | 0.066 | 0.065 |
| 0.500 | 0.179 | 0.083 | 2.177 | 0.115 | 0.090 | 0.085 | 0.083 |
| 1.000 | 0.241 | 0.094 | 2.276 | 0.157 | 0.102 | 0.098 | 0.094 |
| 2.000 | 0.553 | 0.202 | 2.299 | 0.284 | 0.212 | 0.246 | 0.246 |
| 3.000 | 0.631 | 0.212 | 2.260 | 0.368 | 0.235 | 0.304 | 0.304 |
| 4.000 | 0.946 | 0.318 | 2.332 | 0.486 | 0.319 | 0.365 | 0.365 |
| 6.000 | 1.140 | 0.462 | 2.402 | 0.691 | 0.458 | 0.468 | 0.468 |
| 8.000 | 1.357 | 0.679 | 2.301 | 0.905 | 0.605 | 0.544 | 0.544 |
| 12.000 | 1.844 | 1.052 | 2.322 | 1.316 | 0.987 | 0.755 | 0.755 |
| 24.000 | 2.110 | 1.820 | 2.186 | 3.139 | 1.794 | 0.966 | 0.966 |
| 48.000 | 2.632 | 1.993 | 2.704 | 5.789 | 2.034 | 1.446 | 1.446 |
| 72.000 | 3.002 | 2.330 | 2.773 | 7.704 | 2.350 | 1.783 | 1.783 |

## Performance (filling all real gaps of the primary sensor)
| method | seconds | peak_MB | flops | MFLOP_per_s | gaps_fitted |
|---|---|---|---|---|---|
| ffill | 0.0004064 | 0.5204 | nan | nan | nan |
| linear | 0.0004159 | 0.7454 | 6955 | 16.72 | 0 |
| yesterday | 0.0007801 | 1.278 | nan | nan | nan |
| cubic_spline | 0.001526 | 2.482 | 1.243e+05 | 81.45 | 0 |
| pchip | 0.001375 | 1.93 | 2.899e+05 | 210.8 | 0 |
| harmonic | 0.5595 | 4.057 | 5.512e+08 | 985.1 | 1390 |
| hybrid | 0.0008335 | 1.004 | 0 | nan | 0 |

Optimisation of the harmonic method (accuracy check: max difference from the naive version):

| version | seconds | speedup | peak_MB | max_diff_C | sensor |
|---|---|---|---|---|---|
| 1 naive (basis rebuilt per gap) | 3.89 | 1 | 1.004 | 0 | 6012002000869 |
| 2 basis built once | 0.4912 | 7.921 | 4.057 | 1.421e-14 | 6012002000869 |
| 3 + skip short gaps (hybrid) | 0.0005463 | 7121 | 1.004 | 0.0228 | 6012002000869 |
| 1 naive (basis rebuilt per gap) | 6.141 | 1 | 2.16 | 0 | 6012002000326 |
| 2 basis built once | 0.832 | 7.381 | 8.712 | 3.553e-14 | 6012002000326 |
| 3 + skip short gaps (hybrid) | 0.03957 | 155.2 | 8.712 | 0.1013 | 6012002000326 |

Runtime vs number of gaps at fixed length (log-log slope): harmonic 0.97,
hybrid -0.08.

Loop vs vectorised linear interpolation on 1e6 slots: 0.176 s vs 0.0253 s
(speed-up 7x, identical output: True).

Scaling exponents (slope of log time vs log n):

| method | exponent | n_range |
|---|---|---|
| linear | 1.013 | 64000-1024000 |
| yesterday | 1.159 | 64000-1024000 |
| cubic_spline | 1.026 | 64000-1024000 |
| pchip | 0.997 | 64000-1024000 |
| hybrid | 1.001 | 64000-1024000 |
| harmonic | 1.012 | 16000-256000 |

Least-squares solver comparison:

| trend_centred | solver | cond_A | cond_AtA | seconds | max_coef_diff_vs_svd | rows_m | params_p |
|---|---|---|---|---|---|---|---|
| False | lstsq | 2.42e+04 | 5.83e+08 | 0.000181 | 0 | 672 | 14 |
| False | normal | 2.42e+04 | 5.83e+08 | 3.58e-05 | 1.25e-10 | 672 | 14 |
| True | lstsq | 2.52 | 6.37 | 0.000181 | 0 | 672 | 14 |
| True | normal | 2.52 | 6.37 | 3.55e-05 | 2.84e-14 | 672 | 14 |

## Reconstruction of real gaps (deployed method: hybrid)
| sensor | filled_slots | gaps | longest_gap_h | slots_beyond_validated_range |
|---|---|---|---|---|
| 6012002000869 | 1391 | 1390 | 0.300 | 0 |
| 6012002000326 | 3669 | 2143 | 107.700 | 1236 |
