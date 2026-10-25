# Reconstruction of Missing Indoor Temperature Readings

MMA3001 Numerical Methods and Machine Learning - individual project.
Dataset 2: Monash Smart Infrastructure environmental sensor data.

This project fills gaps in 10-minute indoor temperature records from NEMo
environmental sensors and answers one practical question: **how long a gap
can be filled before the reconstructed values become less trustworthy than
the sensor itself?** It compares numerical interpolation (linear, cubic
spline, PCHIP) and a harmonic least-squares model against simple baselines,
validated on artificial gaps with known ground truth.

![Error vs gap length](results/figures/03_error_vs_gap_test.png)

**Headline result.** On the primary sensor's held-out data, gaps of up to
1 hour are filled to about 0.1 °C by plain linear interpolation, and no
method does measurably better. For longer gaps, harmonic least squares,
which models the room's daily cycle, halves the error of interpolation
(12 h: 0.42 vs 0.85 °C) and stays within the sensor's ±0.5 °C accuracy up
to the longest gap tested (12 h), against 4 h for linear interpolation. On
a stricter worst-case check, and on a second, busier room, the reliable
limit is about 4 hours.

---

## 1. Engineering problem

**Context.** Monash Buildings 60 and 69 carry environmental sensors that
report temperature, humidity, CO2 and particulates every 10 minutes. These
records support thermal-comfort assessment, HVAC scheduling and fault
detection, and energy analysis.

**Problem.** Readings go missing: 9% of 10-minute slots for the primary
sensor and 11% for the secondary one, mostly single dropped uploads, plus
two multi-day outages on the secondary sensor. Most analyses need a
continuous record, and naive filling can bias results, for example by
hiding an afternoon temperature peak in a comfort assessment.

**Intended use.** *After-the-fact* reconstruction of historical records
(comfort audits, energy reports), where readings on both sides of a gap
are available. It is not intended for live monitoring during an outage;
there, only forward fill is possible.

**Scope.**
* **Primary problem: short gaps** (10 minutes to 2 hours), the sensor's
  real dropout pattern, studied on sensor 0869.
* **Secondary: how far reconstruction can be pushed**: gaps up to 12 hours
  on sensor 0869, and a long-gap case study up to 3 days on sensor 0326,
  including its two real 4-day outages.
* Temperature only, one sensor at a time. The two usable sensors never
  overlap in time, so neighbouring-room information cannot be used.
* Outages longer than 7 days split a record into separate spans, and only
  the longest span is reconstructed.

## 2. Inputs and outputs

### Inputs

| Input | Format | Units / domain | Handling of invalid data |
|---|---|---|---|
| `data/5EnvSensor_MayToDec2024_180kRows.zip` | ZIP containing one CSV with columns `sensorid` (string) and `jsondata` (JSON list of `{variable, values}` objects) | Temperature in °C (variable `"Temperature"`, unit `"°C"`); time as Unix seconds (UTC) | Malformed JSON, missing IDs, non-numeric values and invalid times are counted and dropped; other units are ignored; duplicate uploads are averaged (conflicts counted: none found); values outside 10-40 °C are flagged `suspect` (none found) |
| `data/Sensor ID and Locations.xlsx` (optional) | Workbook, sheet `ENV` | Building, room, area per sensor ID | If missing, locations show as unknown |

The dataset belongs to Monash University and is **not** in this repository.
Download it from Moodle and place both files in `data/` with these names.

### Outputs (in `results/`)

| Output | Meaning and intended use |
|---|---|
| `reconstructed_<sensor>.csv` | Continuous 10-minute series: `time` (Australia/Melbourne), `temperature` (°C, measured or filled), `reconstructed` (bool), `gap_len_slots` (0 if measured), `expected_abs_error` (validated MAE for that gap length on that sensor, °C; 0 if measured; NaN beyond the longest tested gap, where the fill is unvalidated) |
| `RESULTS.md` | Auto-generated summary of every headline number |
| `selected_settings.json` | Settings and decisions made on the development data |
| `tables/*.csv` | Raw results of every experiment |
| `figures/*.png` | Report figures |
| `test_report.html` | pytest report |

Function-level inputs and outputs are documented in docstrings and in the
HTML documentation (`docs/index.html`).

## 3. Data audit

| Finding | Consequence |
|---|---|
| Only 2 of the 5 sensors have usable data; the other three have 1-2 readings each | **0869** (B60 G.38 Compactus, upper; Feb-Jun 2024; 13,800 readings) and **0326** (location not in the table; Sep 2025-Apr 2026; 29,038 readings) |
| 76% of CSV rows are exact duplicates, with no conflicting values | Deduplicated by timestamp |
| The file name says "May to Dec 2024", but measurement times span 2023-2026 | Measurement time taken from `jsondata`, not `createdate` |
| Readings are not on clock 10-minute marks: 0869 reports 4:55-5:56 past each mark; 0326's offset drifts through the whole 10 minutes | Slot assignment must be drift-robust (decision 3) |
| Over 95% of gaps are a single missed reading; the longest fully observed stretch is 11 h (0869) and 4.7 h (0326) | Shapes the validation design (decision 4) |

![Sensor overview](results/figures/01_sensor_overview.png)

## 4. Design decisions

Each decision was made with the alternatives in view. The reasoning is
summarised here so that every choice can be traced.

| # | Decision | Alternatives considered | Reason |
|---|---|---|---|
| 1 | **Primary sensor 0869; 0326 as an independent check** | 0326 primary; both combined | 0869 has a known location (a storage room), so results can be interpreted physically. 0326, a different room and year with more variable temperature, is a deliberately harder robustness test. |
| 2 | **Short gaps primary; long gaps as a secondary study** | Short only; short and long equally | Matches the sensor's real dropout pattern, while still finding where reconstruction breaks down. |
| 3 | **Consecutive-interval slot assignment**: `slot[i] = slot[i-1] + round(dt/10 min)` | Round to clock time; fixed bins; elapsed time from the first reading | Clock rounding splits 0869's readings, which sit right at the 5-minute rounding point. Fixed bins fail under 0326's drift. Measuring from the first reading accumulates drift. Rounding each interval absorbs slow drift: 0 collisions, maximum timing error 27 s (0869) and 77 s (0326). |
| 4 | **Artificial gaps in two designs**: fully observed (every slot has a true value) at 10, 30, 60, 120 min; partially observed (>= 80%, scored only where the truth is known) at 1-12 h | Leave-one-out; visual inspection | Real gaps have no ground truth. Fully observed windows longer than about 2 h are rare (about 3% of 6 h windows), so long gaps need the partial design. Both designs are run at 1 and 2 h to check they agree. |
| 5 | **MAE primary**; RMSE, maximum, 95th-percentile error and bias secondary; 95% bootstrap CIs resampling whole gaps | RMSE or maximum alone | MAE is directly interpretable in °C. Bias reveals systematic shaving of peaks. CIs show whether differences are real; points within a gap are correlated, so whole gaps are resampled. |
| 6 | **Acceptance: MAE <= 0.5 °C**; **usable horizon** = longest gap whose *upper* 95% CI of MAE is <= 0.5 °C at it and every shorter length; secondary check: 95th-percentile error <= 1.0 °C | Arbitrary threshold | ±0.5 °C is the NEMo's calibrated accuracy ([spec](https://www.instrumentchoice.com.au/nemo-air-quality-station)). A reconstruction within it is as trustworthy as a measurement. The sensor's ±2 °C uncalibrated offset cancels, because reconstructions are compared with the sensor's own readings. |
| 7 | **Use case: after-the-fact reconstruction** | Live gap filling | Interpolation needs a reading after the gap. |
| 8 | **Baselines: forward fill, linear, same time yesterday** | Overall mean | Forward fill is what a dashboard does. Linear is the simplest numerical method. Yesterday is the simplest method that uses the daily pattern, and tests whether that pattern carries information. |
| 9 | **Methods: cubic spline, PCHIP, harmonic least squares** | Interpolation only; machine learning; physics-based RC model | Harmonic least squares adds the daily cycle, which interpolation cannot see, and stays within numerical linear algebra. ML and RC models were judged too broad for the scope. |
| 10 | **First 60% of 0869 for every decision; last 40% held out and used once; 0326 never used for decisions** | One data set; blocked cross-validation | Choosing settings on the data you report on flatters the result. Both test sets are untouched by any decision. |
| 11 | **Hybrid rule**: linear up to X, harmonic above; X = largest gap length where linear is within 2% of harmonic at every length up to X; hybrid adopted only if it beats the best single method by more than 0.08 °C (sensor resolution) somewhere without losing by more than 0.08 °C anywhere | Averaging MAE over lengths | Checking each length separately stops a loss at one length hiding in an average. The physical margin stops noise deciding. |
| 12 | **Performance: optimise the harmonic method** (basis built once vs a naive reference; skip short gaps; centred trend; normal equations measured but rejected); scaling with series length and with number of gaps, against a flop cost model | Timing only | The harmonic method is the only bottleneck (about 1,000x slower than interpolation). |

## 5. Methods

All methods take the regular 10-minute series and fill every missing slot.
Observed values are never changed.

| Method | How it works | Assumption / known weakness |
|---|---|---|
| Forward fill | Repeat the last reading | Temperature constant during the gap |
| Linear | Straight line between the readings either side (`numpy.interp`); error O(h²) | No daily cycle inside the gap |
| Same time yesterday | Reading 24 h earlier (up to 7 days back if missing) | Blind to day-to-day level changes and to weekday vs weekend |
| Cubic spline | Global C² spline through all readings, from one banded linear solve; error O(h⁴) | Overshoots on long gaps |
| PCHIP | Piecewise cubic Hermite with Fritsch-Carlson slopes; never overshoots its neighbours | No daily cycle inside the gap |
| Harmonic least squares | For each gap: fit constant + trend + K daily + J weekly sine/cosine pairs to the readings within ±W days by least squares (SVD), then add a straight-line correction so the curve meets the readings at both gap edges | Assumes the daily cycle repeats |
| Hybrid | Linear for gaps up to X, harmonic above | X chosen on development data |

## 6. Results

### Decisions made on the development data (first 60% of 0869)

* **Harmonic settings:** K = 6 daily harmonics, J = 0 weekly, W = 3 days.
  * K: error fell steadily from K = 0 (mean 0.440 °C) to K = 6 (0.286 °C).
    **K = 6 is the edge of the tested range**, so more harmonics might help
    further (see limitations).
  * J: weekly harmonics made **no difference** (0.286 °C for every J).
    Plausible for a storage room with no weekday occupancy pattern.
  * W: flat from 2 to 21 days (0.281-0.287 °C). The choice of 3 days has
    almost no effect.
* **Switch:** linear is within 2% of harmonic up to 1 hour, so X = 1 h.
* **Hybrid not adopted on accuracy:** its best gain over the harmonic
  model (the best single method) is under 0.08 °C. **Final method:
  harmonic least squares.**
* **Deployed as the hybrid implementation**, because its accuracy is
  within 2% of harmonic at every length on the development data and it is
  far faster (Section 7). In effect, the hybrid is an optimisation of the
  harmonic method rather than a separate method.

### Held-out test MAE (°C), sensor 0869

| Gap | Fwd fill | Linear | Yesterday | Spline | PCHIP | Harmonic | Hybrid |
|---|---|---|---|---|---|---|---|
| 10 min | 0.059 | **0.056** | 0.416 | 0.075 | 0.057 | 0.058 | **0.056** |
| 1 h | 0.212 | 0.104 | 0.394 | 0.144 | **0.100** | 0.104 | 0.104 |
| 2 h | 0.390 | 0.197 | 0.412 | 0.231 | **0.177** | 0.188 | 0.188 |
| 4 h | 0.701 | 0.358 | 0.454 | 0.491 | 0.328 | **0.268** | **0.268** |
| 6 h | 0.862 | 0.481 | 0.425 | 0.684 | 0.446 | **0.315** | **0.315** |
| 12 h | 1.133 | 0.854 | 0.433 | 1.427 | 0.833 | **0.417** | **0.417** |

Up to 2 h: fully observed gaps; above: partially observed. All nine gap
lengths are in `results/RESULTS.md`.

### Usable horizon (upper 95% CI of MAE <= 0.5 °C)

| Method | 0869 (test) | 0326 (independent) |
|---|---|---|
| Forward fill | 2 h | 1 h |
| Linear | 4 h | 4 h |
| Same time yesterday | 6 h | 0 (fails at 10 min) |
| Cubic spline | 3 h | 3 h |
| PCHIP | 6 h | 4 h |
| **Harmonic / hybrid** | **12 h** (longest tested) | **4 h** |

### Findings

1. **Short gaps (up to 1 h): linear is as good as anything.** Linear,
   PCHIP, harmonic and hybrid differ by at most 0.005 °C at 10 minutes,
   far below the 0.08 °C sensor resolution and within the confidence
   intervals. The evidence-based choice for short gaps is the simplest
   method, which is what the hybrid uses.
2. **Long gaps: modelling the daily cycle halves the error.** From 3 h
   the harmonic model is best (12 h: 0.42 vs 0.85 °C for linear) and is
   the only method within 0.5 °C at every tested length.
3. **The daily pattern carries real information.** On 0869, "same time
   yesterday" is poor for short gaps (about 0.4 °C) because it ignores the
   gap edges, but it beats linear from 6 hours. That is the evidence that
   motivated the harmonic model, which combines both sources (daily shape
   plus edge readings).
4. **The cubic spline is the worst interpolant for long gaps** (12 h:
   1.43 °C): an unconstrained cubic between distant knots overshoots. In
   the example gap it dips about 1.7 °C below the true minimum. PCHIP's
   no-overshoot property makes it slightly better than linear.
5. **The worst-case check is stricter than the average.** The harmonic
   model's 95th-percentile error passes 1.0 °C up to 4 h (0.94 °C) but not
   at 6 h (1.03 °C). Average accuracy holds to 12 h, but occasional
   errors above 1 °C appear from 6 h.
6. **Results only partly transfer to the second room.** On 0326 the
   harmonic model helps only from about 8 h. At 2-6 h, linear and PCHIP
   are better (3 h: 0.21 vs 0.30 °C), and every method's horizon is 4 h.
   "Same time yesterday" fails completely (about 2.3 °C) because that
   room's level changes from day to day. **The best switch point between
   linear and harmonic depends on the room.**
7. **The two validation designs agree.** At 1 h and 2 h the fully and
   partially observed MAE confidence intervals overlap for all 14 method
   and length pairs (figure 03c), supporting the use of partially observed
   gaps for longer lengths.
8. **Gap timing matters.** 6-hour gaps starting between midnight and 6 am
   (covering the overnight cool-down and morning warm-up) give linear
   interpolation errors of 0.75-0.9 °C, against about 0.2 °C for gaps
   starting around midday. The harmonic model roughly halves those
   overnight errors (figure 07).

![Example gap](results/figures/04_example_gap.png)

## 7. Performance and optimisation

Filling all 1,390 real gaps of sensor 0869 (times vary about ±30% between
runs; exact values in `results/RESULTS.md`):

| Method | Time | Peak memory | Estimated flops | Achieved rate |
|---|---|---|---|---|
| Linear | ~0.4 ms | 0.7 MB | 7.0e3 | ~17-21 MFLOP/s |
| Cubic spline | ~1.5 ms | 2.5 MB | 1.2e5 | ~80 MFLOP/s |
| PCHIP | ~1.4 ms | 1.9 MB | 2.9e5 | ~210 MFLOP/s |
| Harmonic (all gaps) | ~560 ms | 4.1 MB | 5.5e8 | ~1 GFLOP/s |
| **Hybrid** | **~0.8 ms** | 1.0 MB | 0 (no gap fitted) | - |

**Optimisation steps for the harmonic method** (max difference = largest
deviation from the naive version's output):

| Version | 0869 time | 0326 time | Speed-up (0869 / 0326) | Max difference |
|---|---|---|---|---|
| 1. Naive: sine/cosine basis rebuilt for every gap | 3.9 s | 6.1 s | 1x | 0 |
| 2. Basis built once, sliced per gap | 0.49 s | 0.83 s | 8x / 7x | 4e-14 °C (identical) |
| 3. + skip gaps ≤ 1 h (hybrid) | 0.5 ms | 40 ms | ~7,000x / ~155x | 0.02 / 0.10 °C |

* **Step 2** removes repeated trig evaluation with no change in output.
* **Step 3** removes almost every least-squares fit. Every real gap in
  0869 is 1-2 slots, so no fits remain at all; 0326 still has 15 longer
  gaps to fit. Outputs change only on short gaps, where validation shows
  linear and harmonic are equally accurate.
* **Cost model confirmed.** At a fixed series length (64,000 slots), the
  harmonic method's time grows with the number of gaps with log-log slope
  **0.97**, as predicted by one fit per gap. The hybrid's slope is about 0:
  extra single dropouts cost it nothing (figure 09b). All methods scale
  linearly with series length (slopes 1.00-1.16, figure 09).
* **Arithmetic intensity.** Each harmonic fit is an m x p least-squares
  problem (m ≈ 860 readings in a ±3-day window, p = 14), about 2mp² flops
  on 8mp bytes: roughly p/4 ≈ 3.5 flops per byte. The matrices fit in
  cache, so the achieved ~1 GFLOP/s, far below CPU peak, is set by Python
  overhead per small solve, not by arithmetic or memory bandwidth. The
  interpolants do so little arithmetic that call overhead dominates them
  too (17-210 MFLOP/s).
* **Numerical stability.** Uncentred, the trend column makes the design
  matrix's condition number 2.4×10⁴, and the normal equations' 5.8×10⁸
  (squared), giving a 1.3×10⁻¹⁰ coefficient error against the SVD
  solution. Centring the trend cuts these to 2.5 and 6.4. Normal equations
  are about 5x faster per solve, but after step 3 almost no solves remain,
  so the speed gain is negligible and the stability risk is not worth it:
  **SVD with a centred trend is kept.**
* **Vectorisation:** `numpy.interp` vs an equivalent pure-Python loop on
  10⁶ slots: about 7x faster, identical output.

## 8. Limitations

1. **Validation assumes readings go missing completely at random.**
   Artificial gaps are placed randomly, but real outages may coincide with
   unusual conditions; a power cut that stops a sensor may also stop the
   HVAC. Accuracy on real outages, especially long ones, may therefore be
   worse than measured.
2. **Limited generalisability.** The primary sensor is one storage room
   over one season (late summer to early winter), and the second room
   behaved differently (Finding 6). Peak summer and mid-winter conditions
   are untested, and 0326's room is unknown.
3. **No method can recover events it never saw.** A door left open, an
   HVAC fault or a crowded meeting during a gap is invisible to any method
   that uses only this sensor's surrounding readings. This sets a floor on
   long-gap error.
4. **The acceptance criterion's source.** ±0.5 °C comes from a reseller's
   specification and assumes calibration; it should be confirmed for
   Monash's units.
5. **Tuning range.** The selected K = 6 is at the edge of the tested grid,
   so the harmonic model may be slightly under-tuned.
6. **Real outages beyond 3 days** (0326's 4.1- and 4.5-day outages) are
   filled but unvalidated: 1,236 slots carry `expected_abs_error = NaN`.

## 9. Future work

The main limit on long-gap accuracy is missing information, not the
algorithm. The first step would be to obtain **a data extract with
simultaneous readings from neighbouring rooms**. Other variables from the
*same* sensor (humidity, CO2) cannot help, because they arrive in the same
upload and go missing together. **Outdoor temperature** from the Bureau of
Meteorology is available now and would tell the model about weather-driven
changes. Other steps: extending the harmonic tuning grid beyond K = 6, and
choosing the hybrid switch per room.

## 10. Repository structure

```
MMA3001-Temperature-Reconstruction/
├── main.py                 command-line entry point (all stages)
├── tempfill/               the Python package
│   ├── data.py             loading, deduplication, 10-minute grid
│   ├── methods.py          gap-filling methods
│   ├── validation.py       artificial gaps, metrics, CIs, decision rules
│   ├── performance.py      timing, memory, scaling, solver stability, flops
│   ├── experiments.py      the experimental protocol
│   └── plots.py            report figures
├── tests/                  pytest suite (58 tests)
├── docs/                   HTML documentation (pdoc)
├── results/                tables, figures, RESULTS.md, test report
├── archive/                original Stage 1 exploration script
├── data/                   put the dataset here (not committed)
├── requirements.txt
├── AI_USE.md               log of AI assistance
└── LICENSE                 MIT (code only)
```

## 11. Setup and usage

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows (Command Prompt):  .venv\Scripts\activate
# Windows (PowerShell):      .venv\Scripts\Activate.ps1
#   if PowerShell blocks scripts, run once:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# macOS / Linux:             source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset in `data/` (Section 2), then:

```bash
python main.py              # everything, about 8-10 minutes
python main.py --quick      # smoke run with fewer repeats, about 3 minutes
python main.py audit        # one stage: audit | tune | validate | performance | reconstruct
```

All randomness is seeded (`Config.seed = 2026`), so accuracy results
reproduce exactly. Timings vary by machine.

### Tests

```bash
pytest                                                        # 58 tests, a few seconds
pytest --html=results/test_report.html --self-contained-html  # HTML report
```

The tests verify the implementation against known answers:
* linear and cubic polynomials reproduced exactly;
* **measured convergence orders of 2 (linear) and 4 (cubic spline)**, as
  theory predicts;
* exact recovery of a known daily signal by the harmonic model and by
  "same time yesterday";
* PCHIP's no-overshoot property;
* the naive and optimised harmonic versions agree;
* the decision rules (switch, adoption, usable horizon) on hand-built
  cases;
* the slot assignment under clock drift.

### Documentation

```bash
pdoc tempfill -o docs --docformat google
```

Open `docs/index.html`. Docstrings follow the Google style.

## 12. AI use

This project was developed with substantial assistance from Claude
(Anthropic). I chose the dataset and problem, wrote the initial data
exploration (`archive/stage1_exploration_original.py`), and made the
design decisions in Section 4 through a structured review in which Claude
presented options and data and challenged my reasoning. Claude generated
the implementation, tests and documentation to those decisions. Details,
and the errors found in the AI's output, are in [`AI_USE.md`](AI_USE.md);
the reflection on AI use is in the written report.

## 13. References

* Fritsch, F. N. & Carlson, R. E. (1980). Monotone piecewise cubic
  interpolation. *SIAM Journal on Numerical Analysis*, 17(2), 238-246.
* de Boor, C. (2001). *A Practical Guide to Splines* (rev. ed.). Springer.
* Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.).
  Johns Hopkins University Press. (Least squares, conditioning, flop counts.)
* Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
  Chapman & Hall.
* Little, R. J. A. & Rubin, D. B. (2019). *Statistical Analysis with
  Missing Data* (3rd ed.). Wiley. (Missing-completely-at-random assumption.)
* ASHRAE Standard 55 (2023). *Thermal Environmental Conditions for Human
  Occupancy*.
* Instrument Choice. NEMo Air Quality Station, product specifications.
  https://www.instrumentchoice.com.au/nemo-air-quality-station
* Monash Smart Infrastructure, NEMo environmental sensor export, provided
  for MMA3001 (2026).

## Licence

Code: MIT (see `LICENSE`). Data: Monash University, not redistributed.
