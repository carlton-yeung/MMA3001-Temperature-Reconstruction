"""End-to-end experiments: audit, tuning, validation, performance, reconstruction.

Data protocol (chronological, per primary sensor)::

    |---------- train ----------|-- validation --|------- test -------|
    0%                         55%              70%                 100%
     boosted model fitted here   harmonic tuning   final reported
                                 + method choice   results only

The secondary sensor (different room, different year) is never used for
fitting or tuning; it measures how well the chosen settings generalise.

Every stage writes CSV tables to ``results/tables`` and figures to
``results/figures``; ``write_summary`` collects the key numbers in
``results/RESULTS.md``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import methods as M
from . import performance as P
from . import plots
from .data import (audit_sensors, gap_runs, load_locations, load_temperatures,
                   prepare_sensor)
from .validation import apply_gaps, run_gap_experiment, sample_gaps, summarise


@dataclass
class Config:
    """All settings for a run. Defaults reproduce the reported results."""
    root: Path = Path(__file__).resolve().parent.parent
    data_zip: str = "data/5EnvSensor_MayToDec2024_180kRows.zip"
    locations_xlsx: str = "data/Sensor ID and Locations.xlsx"
    primary: str = "6012002000326"     # longest record; used for train/val/test
    secondary: str = "6012002000869"   # different room and year; generalisation only
    train_end: float = 0.55
    val_end: float = 0.70
    gap_lengths: tuple = (1, 3, 6, 18, 36, 72, 144, 288, 432)  # 10 min ... 3 days
    n_gaps: int = 20
    repeats: int = 10
    tune_repeats: int = 4
    acceptance_mae: float = 0.5   # °C; ASSUMED from typical sensor accuracy - verify vs datasheet
    seed: int = 2026
    quick: bool = False

    def __post_init__(self):
        if self.quick:  # fast smoke run for development and the test suite
            self.repeats, self.tune_repeats, self.n_gaps = 2, 1, 8

    @property
    def tables(self) -> Path:
        return self.root / "results" / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "results" / "figures"


@dataclass
class Context:
    """Data shared between stages (loaded once)."""
    readings: pd.DataFrame
    frames: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    locations: dict = field(default_factory=dict)


def _save_table(df: pd.DataFrame, cfg: Config, name: str) -> pd.DataFrame:
    cfg.tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.tables / f"{name}.csv", index=False)
    return df


def _regions(n: int, cfg: Config):
    a, b = int(n * cfg.train_end), int(n * cfg.val_end)
    return (0, a), (a, b), (b, n)


def _params_path(cfg: Config) -> Path:
    return cfg.root / "results" / "selected_settings.json"


def load_params(cfg: Config) -> dict:
    """Settings chosen by ``tune`` (defaults if tuning has not been run)."""
    p = _params_path(cfg)
    if p.is_file():
        return json.loads(p.read_text())
    return {"k_daily": 3, "k_weekly": 2, "window_days": 7.0, "switch_len": 6,
            "final_method": "hybrid"}


# ----------------------------------------------------------------------------
# Stage 1: audit
# ----------------------------------------------------------------------------
def audit(cfg: Config) -> Context:
    """Load the data, audit every sensor and regularise the two usable ones."""
    readings, raw_counts, load_stats = load_temperatures(cfg.root / cfg.data_zip)
    locations = load_locations(cfg.root / cfg.locations_xlsx)
    table = audit_sensors(readings, raw_counts, locations)
    _save_table(table.assign(start=table["start"].astype(str), end=table["end"].astype(str)),
                cfg, "sensor_audit")

    ctx = Context(readings=readings, locations=locations, stats={"load": load_stats})
    for sid in (cfg.primary, cfg.secondary):
        frame, st = prepare_sensor(readings, sid)
        ctx.frames[sid] = frame
        ctx.stats[sid] = st
    _save_table(pd.DataFrame([{"sensor_id": k, **v} for k, v in ctx.stats.items() if k != "load"]),
                cfg, "prepared_series")

    named = {f"Sensor {sid} ({locations.get(sid, 'location unknown')})": f
             for sid, f in ctx.frames.items()}
    plots.sensor_overview(named, cfg.figures / "01_sensor_overview.png")
    plots.gap_length_histogram(named, cfg.figures / "02_gap_lengths.png")
    return ctx


# ----------------------------------------------------------------------------
# Stage 2: tuning and method selection (validation region only)
# ----------------------------------------------------------------------------
def _sweep(frame, region, lengths, configs: dict, cfg: Config) -> pd.DataFrame:
    """Evaluate several method configurations on the same synthetic gaps."""
    per_gap = run_gap_experiment(frame, configs, lengths, cfg.n_gaps, cfg.tune_repeats,
                                 region, cfg.seed)
    return summarise(per_gap)


def tune(cfg: Config, ctx: Context) -> dict:
    """Choose harmonic settings, the hybrid switch length and the final method.

    All decisions use only the validation region of the primary sensor.
    Long-gap sweeps use ``min_len=6`` so short real gaps are not refitted
    each time (it does not change results at the gap lengths swept).
    """
    frame = ctx.frames[cfg.primary]
    train, val, _ = _regions(len(frame), cfg)
    long_lengths = (36, 144, 432)

    def harm(**kw):
        return lambda f: M.fill_harmonic(f, min_len=6, **kw)

    # 1. Number of daily harmonics.
    k_tab = pd.concat([
        _sweep(frame, val, long_lengths, {k: harm(k_daily=k)}, cfg).assign(k_daily=k)
        for k in (0, 1, 2, 3, 4, 6)], ignore_index=True)
    best_k = int(k_tab.groupby("k_daily")["MAE"].mean().idxmin())

    # 2. Number of weekly harmonics.
    kw_tab = pd.concat([
        _sweep(frame, val, long_lengths, {j: harm(k_daily=best_k, k_weekly=j)}, cfg)
        .assign(k_weekly=j) for j in (0, 1, 2, 3)], ignore_index=True)
    best_kw = int(kw_tab.groupby("k_weekly")["MAE"].mean().idxmin())

    # 3. Fitting-window half-width.
    w_tab = pd.concat([
        _sweep(frame, val, long_lengths,
               {w: harm(k_daily=best_k, k_weekly=best_kw, window_days=w)}, cfg)
        .assign(window_days=w) for w in (2, 3, 5, 7, 10, 14, 21)], ignore_index=True)
    best_w = float(w_tab.groupby("window_days")["MAE"].mean().idxmin())
    hk = dict(k_daily=best_k, k_weekly=best_kw, window_days=best_w)

    # 4. Hybrid switch length: the largest value whose MAE is within 2% of
    #    the best at *every* gap length (larger = fewer fits = faster).
    #    Checking each length separately stops a real loss at one length
    #    being hidden by averaging.
    short_lengths = (1, 3, 6, 12, 18, 36)
    s_tab = pd.concat([
        _sweep(frame, val, short_lengths,
               {s: (lambda f, s=s: M.fill_hybrid(f, switch_len=s, **hk))}, cfg)
        .assign(switch_len=s) for s in (0, 1, 2, 3, 6, 12, 18)], ignore_index=True)
    s_tab["ratio"] = s_tab["MAE"] / s_tab.groupby("gap_len")["MAE"].transform("min")
    worst = s_tab.groupby("switch_len")["ratio"].max()
    within = worst[worst <= 1.02]
    # Fall back to the most consistent switch if none is within tolerance everywhere.
    best_switch = int(within.index.max()) if len(within) else int(worst.idxmin())

    for name, tab in [("tune_k_daily", k_tab), ("tune_k_weekly", kw_tab),
                      ("tune_window", w_tab), ("tune_switch", s_tab)]:
        _save_table(tab.drop(columns="method"), cfg, name)
    plots.sensitivity(k_tab, "k_daily", cfg.figures / "05_sens_k_daily.png",
                      "Daily harmonics (validation period)", "Number of daily harmonics K")
    plots.sensitivity(w_tab, "window_days", cfg.figures / "06_sens_window.png",
                      "Fitting window (validation period)", "Window half-width (days)")

    # 5. Method selection on the validation region.
    boosted = M.BoostedFill(**hk, random_state=cfg.seed).fit(frame, train)
    candidates = _method_set(hk, best_switch, boosted)
    sel = summarise(run_gap_experiment(frame, candidates, cfg.gap_lengths, cfg.n_gaps,
                                       cfg.tune_repeats, val, cfg.seed + 1))
    _save_table(sel, cfg, "method_selection_validation")
    score = sel.groupby("method").agg(mean_MAE=("MAE", "mean"), seconds=("seconds", "mean"))
    near_best = score[score["mean_MAE"] <= 1.02 * score["mean_MAE"].min()]
    final = str(near_best["seconds"].idxmin())  # fastest within 2% of the best

    params = {**hk, "switch_len": best_switch, "final_method": final,
              "selection_scores": score.round(4).reset_index().to_dict("records")}
    _params_path(cfg).write_text(json.dumps(params, indent=2))
    return params


def _method_set(hk: dict, switch_len: int, boosted) -> dict:
    """All compared methods, configured with the tuned harmonic settings."""
    methods = {
        "ffill": M.fill_ffill, "linear": M.fill_linear,
        "cubic_spline": M.fill_cubic_spline, "pchip": M.fill_pchip,
        "harmonic": lambda f: M.fill_harmonic(f, **hk),
        "hybrid": lambda f: M.fill_hybrid(f, switch_len=switch_len, **hk),
    }
    if boosted is not None:
        methods["boosted"] = boosted
    return methods


def _harmonic_kw(params: dict) -> dict:
    return {k: params[k] for k in ("k_daily", "k_weekly", "window_days")}


# ----------------------------------------------------------------------------
# Stage 3: held-out validation
# ----------------------------------------------------------------------------
def validate(cfg: Config, ctx: Context, params: dict) -> dict:
    """Final comparison on the held-out test period and on the second sensor."""
    frame = ctx.frames[cfg.primary]
    train, _, test = _regions(len(frame), cfg)
    hk = _harmonic_kw(params)
    boosted = M.BoostedFill(**hk, random_state=cfg.seed).fit(frame, train)
    methods = _method_set(hk, params["switch_len"], boosted)

    per_gap = run_gap_experiment(frame, methods, cfg.gap_lengths, cfg.n_gaps, cfg.repeats,
                                 test, cfg.seed + 2)
    summary = _save_table(summarise(per_gap), cfg, "test_results_primary")
    _save_table(per_gap, cfg, "test_per_gap_primary")
    plots.error_vs_gap(summary, cfg.figures / "03_error_vs_gap_test.png",
                       f"Held-out test period, sensor {cfg.primary}", cfg.acceptance_mae)
    plots.error_vs_gap(summary, cfg.figures / "03b_maxerror_vs_gap_test.png",
                       f"Worst-case error, sensor {cfg.primary}", metric="MaxAE")
    plots.error_by_hour(per_gap, frame, 36, ["linear", "harmonic", "boosted"],
                        cfg.figures / "07_error_by_start_hour.png")

    # Example 1-day gap from the test period.
    rng = np.random.default_rng(cfg.seed)
    obs = frame["temperature"].notna().to_numpy()
    start = sample_gaps(obs, 144, 1, rng, test)[0]
    masked, _ = apply_gaps(frame, [start], 144)
    fills = {m: methods[m](masked) for m in ("linear", "cubic_spline", "harmonic", "boosted")}
    plots.example_gap(frame, start, 144, fills, cfg.figures / "04_example_gap.png")

    # Generalisation: second sensor, never used for fitting or tuning.
    f2 = ctx.frames[cfg.secondary]
    per_gap2 = run_gap_experiment(f2, methods, cfg.gap_lengths, cfg.n_gaps,
                                  max(2, cfg.repeats // 2), None, cfg.seed + 3)
    summary2 = _save_table(summarise(per_gap2), cfg, "test_results_secondary")
    plots.error_vs_gap(summary2, cfg.figures / "08_error_vs_gap_secondary.png",
                       f"Generalisation: sensor {cfg.secondary} (unseen room and year)",
                       cfg.acceptance_mae)

    # Feature importance proxy for the boosted model is not built in for
    # HistGradientBoosting; record training size instead.
    return {"primary": summary, "secondary": summary2,
            "boosted_rows": boosted.n_train_rows_, "boosted_iters": int(boosted.model.n_iter_)}


# ----------------------------------------------------------------------------
# Stage 4: performance
# ----------------------------------------------------------------------------
def performance(cfg: Config, ctx: Context, params: dict) -> dict:
    """Runtime, memory, scaling, solver stability and profiling."""
    frame = ctx.frames[cfg.primary]
    hk = _harmonic_kw(params)
    methods = _method_set(hk, params["switch_len"], None)
    reps = 1 if cfg.quick else 3

    costs = P.method_costs(frame, methods, repeats=reps)
    flops = P.flop_estimates(frame, switch_len=params["switch_len"], **hk)
    costs = costs.merge(flops, on="method", how="left")
    costs["MFLOP_per_s"] = costs["flops"] / costs["seconds"] / 1e6
    _save_table(costs, cfg, "perf_method_costs")

    train, _, _ = _regions(len(frame), cfg)
    import time as _t
    t0 = _t.perf_counter()
    boosted = M.BoostedFill(**hk, random_state=cfg.seed).fit(frame, train)
    fit_s = _t.perf_counter() - t0
    boost_cost = {"fit_seconds": fit_s, "fill_seconds": P.time_call(boosted, frame, repeats=reps),
                  "train_rows": boosted.n_train_rows_}

    big = P.synthetic_frame(100_000 if cfg.quick else 1_000_000)
    vec = P.vectorisation_speedup(big, M.fill_linear_loop, M.fill_linear, repeats=reps)
    solver = _save_table(P.solver_comparison(frame, **hk), cfg, "perf_solver_comparison")

    sizes = (4_000, 16_000, 64_000) if cfg.quick else (4_000, 16_000, 64_000, 256_000, 1_024_000)
    scale_methods = {k: methods[k] for k in ("linear", "cubic_spline", "pchip", "hybrid", "harmonic")}
    times, exps = P.scaling(scale_methods, sizes, repeats=reps,
                            max_n={"harmonic": 64_000 if cfg.quick else 256_000})
    _save_table(times, cfg, "perf_scaling")
    _save_table(exps, cfg, "perf_scaling_exponents")
    plots.scaling_plot(times, exps, cfg.figures / "09_scaling.png")

    prof = "\n".join([
        "=== harmonic (all gaps) ===", P.profile_call(methods["harmonic"], frame),
        "=== hybrid ===", P.profile_call(methods["hybrid"], frame)])
    (cfg.root / "results" / "profile_harmonic.txt").write_text(prof)
    out = {"vectorisation": vec, "boosted": boost_cost}
    (cfg.root / "results" / "perf_misc.json").write_text(json.dumps(out, indent=2))
    return {"costs": costs, "solver": solver, "exponents": exps, **out}


# ----------------------------------------------------------------------------
# Stage 5: reconstruct real gaps
# ----------------------------------------------------------------------------
def reconstruct(cfg: Config, ctx: Context, params: dict,
                test_summary: pd.DataFrame | None = None) -> dict:
    """Fill the real gaps of both sensors with the selected method.

    Output columns: ``time``, ``temperature`` (measured or reconstructed),
    ``reconstructed`` (bool), ``gap_len_slots`` (0 if measured) and
    ``expected_abs_error`` - the validated MAE for that gap length,
    interpolated on log gap length (NaN beyond the tested range, where the
    reconstruction is an unvalidated extrapolation).
    """
    hk = _harmonic_kw(params)
    name = params["final_method"]
    frame0 = ctx.frames[cfg.primary]
    train, _, _ = _regions(len(frame0), cfg)
    boosted = (M.BoostedFill(**hk, random_state=cfg.seed).fit(frame0, train)
               if name == "boosted" else None)
    method = _method_set(hk, params["switch_len"], boosted)[name]
    if test_summary is None and (cfg.tables / "test_results_primary.csv").is_file():
        test_summary = pd.read_csv(cfg.tables / "test_results_primary.csv")

    out = {}
    for sid, frame in ctx.frames.items():
        filled = method(frame)
        missing = frame["temperature"].isna().to_numpy()
        gap_len = np.zeros(len(frame), dtype=int)
        starts, lengths = gap_runs(missing)
        for s, L in zip(starts, lengths):
            gap_len[s:s + L] = L
        expected = np.full(len(frame), np.nan)
        expected[~missing] = 0.0
        if test_summary is not None:
            g = test_summary[test_summary["method"] == name].sort_values("gap_len")
            x = np.log(g["gap_len"].to_numpy())
            inside = missing & (gap_len <= g["gap_len"].max())
            expected[inside] = np.interp(np.log(gap_len[inside]), x, g["MAE"].to_numpy())
        result = pd.DataFrame({"time": frame["time"], "temperature": filled,
                               "reconstructed": missing, "gap_len_slots": gap_len,
                               "expected_abs_error": expected})
        cfg.tables.mkdir(parents=True, exist_ok=True)
        result.to_csv(cfg.root / "results" / f"reconstructed_{sid}.csv", index=False)
        out[sid] = {"filled_slots": int(missing.sum()), "longest_gap_slots": int(lengths.max())}
        if lengths.max() >= 144:
            s = starts[np.argmax(lengths)]
            plots.reconstruction(frame, filled, s, int(lengths.max()),
                                 cfg.figures / f"10_reconstruction_{sid}.png", name)
    return out


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Render a DataFrame as a Markdown table (no extra dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(cfg: Config, ctx: Context, params: dict, val: dict, perf: dict,
                  recon: dict) -> Path:
    """Write ``results/RESULTS.md`` with the headline numbers of a run."""
    prim = val["primary"]
    mae = prim.pivot(index="gap_hours", columns="method", values="MAE")
    mae = mae[[m for m in plots.METHOD_ORDER if m in mae.columns]].reset_index()
    mae2 = val["secondary"].pivot(index="gap_hours", columns="method", values="MAE")
    mae2 = mae2[[m for m in plots.METHOD_ORDER if m in mae2.columns]].reset_index()
    final = params["final_method"]
    best = prim.loc[prim.groupby("gap_len")["MAE"].idxmin(), ["gap_hours", "method", "MAE"]]
    ok = prim[(prim["method"] == final) & (prim["MAE"] <= cfg.acceptance_mae)]
    horizon = ok["gap_hours"].max() if len(ok) else 0
    costs = perf["costs"][["method", "seconds", "peak_MB", "flops", "MFLOP_per_s",
                           "gaps_fitted"]]
    vec = perf["vectorisation"]

    text = f"""# Results summary (auto-generated)

Run settings: repeats={cfg.repeats}, gaps per length={cfg.n_gaps}, seed={cfg.seed},
quick={cfg.quick}. Primary sensor {cfg.primary}, secondary sensor {cfg.secondary}.

## Data
- CSV rows: {ctx.stats['load']['csv_rows']}; temperature readings after deduplication:
  {ctx.stats[cfg.primary]['observed']} (primary), {ctx.stats[cfg.secondary]['observed']} (secondary).
- Primary: {ctx.stats[cfg.primary]['slots']} slots, {ctx.stats[cfg.primary]['missing']} missing
  ({ctx.stats[cfg.primary]['missing'] / ctx.stats[cfg.primary]['slots']:.1%}),
  {ctx.stats[cfg.primary]['start']} to {ctx.stats[cfg.primary]['end']}.
- Secondary: {ctx.stats[cfg.secondary]['slots']} slots, {ctx.stats[cfg.secondary]['missing']} missing.

## Settings chosen on the validation period
k_daily={params['k_daily']}, k_weekly={params['k_weekly']}, window_days={params['window_days']},
hybrid switch_len={params['switch_len']} slots. **Selected method: {final}**
(fastest method within 2% of the lowest mean validation MAE).

{_md(pd.DataFrame(params['selection_scores']))}

## Held-out test MAE (°C), primary sensor
{_md(mae)}

Best method per gap length:

{_md(best)}

Selected method meets the {cfg.acceptance_mae} °C acceptance MAE up to {horizon:g} h gaps.

## Generalisation MAE (°C), secondary sensor
{_md(mae2)}

## Performance (filling all real gaps of the primary sensor)
{_md(costs, "{:.4g}")}

Loop vs vectorised linear interpolation: {vec['loop_s']:.3f} s vs {vec['vectorised_s']:.5f} s
(speed-up {vec['speedup']:.0f}x, identical output: {vec['identical_output']}).

Boosted model: fit {perf['boosted']['fit_seconds']:.1f} s on {perf['boosted']['train_rows']} rows;
fill {perf['boosted']['fill_seconds']:.2f} s.

Scaling exponents (slope of log time vs log n):

{_md(perf['exponents'])}

Least-squares solver comparison:

{_md(perf['solver'], "{:.3g}")}

## Reconstruction of real gaps
{_md(pd.DataFrame([{'sensor': k, **v} for k, v in recon.items()]))}
"""
    path = cfg.root / "results" / "RESULTS.md"
    path.write_text(text, encoding="utf-8")
    return path
