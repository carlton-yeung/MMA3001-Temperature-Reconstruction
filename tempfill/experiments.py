"""End-to-end experiments: audit, tuning, validation, performance, reconstruction.

Data protocol for the primary sensor (0869, chronological)::

    |----------- development (60%) -----------|------ test (40%) ------|
     harmonic tuning, method comparison,         used once, after every
     hybrid switch and adoption decisions        decision is frozen

The secondary sensor (0326: different room, different year) is never used
for any decision. It is a second independent test, and the case study for
long gaps (up to 3 days) and its real multi-day outages.

Two artificial-gap designs (``validation.sample_gaps``):

* **fully observed** - every slot in the gap holds a real reading; used for
  10 min to 2 h, where such windows are common;
* **partially observed** - at least 80% of slots hold real readings and
  only those are scored; used for 1 h and longer, because fully observed
  windows longer than a few hours are rare in this data.

Both designs are run at 1 h and 2 h so their agreement can be checked.
Headline tables use the fully observed results up to 2 h and the partially
observed results above.

Every stage writes CSV tables to ``results/tables`` and figures to
``results/figures``; ``write_summary`` collects the key numbers in
``results/RESULTS.md``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import methods as M
from . import performance as P
from . import plots
from .data import (audit_sensors, gap_runs, load_locations, load_temperatures,
                   prepare_sensor)
from .validation import (adopt_hybrid, apply_gaps, choose_switch, run_gap_experiment,
                         sample_gaps, simplest_within, summarise, usable_horizon)


@dataclass
class Config:
    """All settings for a run. Defaults reproduce the reported results."""
    root: Path = Path(__file__).resolve().parent.parent
    data_zip: str = "data/5EnvSensor_MayToDec2024_180kRows.zip"
    locations_xlsx: str = "data/Sensor ID and Locations.xlsx"
    primary: str = "6012002000869"     # B60 G.38 Compactus (upper); development and test
    secondary: str = "6012002000326"   # location unknown; independent test + case study
    dev_end: float = 0.60              # first 60% for decisions, last 40% held out
    short_lengths: tuple = (1, 3, 6, 12)                 # 10, 30, 60, 120 min - fully observed
    long_lengths: tuple = (6, 12, 18, 24, 36, 48, 72)    # 1-12 h - partially observed
    case_lengths: tuple = (6, 12, 18, 24, 36, 48, 72, 144, 288, 432)  # secondary, up to 3 days
    n_gaps: int = 20
    repeats_short: int = 10
    repeats_long: int = 20             # more repeats: few long gaps fit per placement
    tune_repeats: int = 4
    acceptance_mae: float = 0.5        # °C, NEMo calibrated accuracy (+/-0.5 °C)
    p95_limit: float = 1.0             # °C, secondary worst-case check
    resolution: float = 0.08           # °C, NEMo resolution: hybrid adoption margin
    switch_tolerance: float = 0.02     # linear must be within 2% of harmonic below the switch
    seed: int = 2026
    quick: bool = False

    def __post_init__(self):
        if self.quick:  # fast smoke run for development
            self.repeats_short, self.repeats_long, self.tune_repeats, self.n_gaps = 2, 2, 1, 8

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
    a = int(n * cfg.dev_end)
    return (0, a), (a, n)


def _params_path(cfg: Config) -> Path:
    return cfg.root / "results" / "selected_settings.json"


def load_params(cfg: Config) -> dict:
    """Settings chosen by ``tune`` (defaults if tuning has not been run)."""
    p = _params_path(cfg)
    if p.is_file():
        return json.loads(p.read_text())
    return {"k_daily": 2, "k_weekly": 2, "window_days": 7.0, "switch_len": 6,
            "final_method": "harmonic", "deployed": "hybrid"}


def _harmonic_kw(params: dict) -> dict:
    return {k: params[k] for k in ("k_daily", "k_weekly", "window_days")}


def method_set(hk: dict, switch_len: int | None = None) -> dict:
    """All compared methods; harmonic uses the tuned settings ``hk``."""
    methods = {
        "ffill": M.fill_ffill, "linear": M.fill_linear, "yesterday": M.fill_yesterday,
        "cubic_spline": M.fill_cubic_spline, "pchip": M.fill_pchip,
        "harmonic": lambda f: M.fill_harmonic(f, **hk),
    }
    if switch_len is not None:
        methods["hybrid"] = lambda f: M.fill_hybrid(f, switch_len=switch_len, **hk)
    return methods


def evaluate(frame, methods, region, cfg: Config, seed: int, long_lengths=None,
             repeats_long=None) -> dict:
    """Run both gap designs and build the headline (combined) summary.

    Returns:
        Dict with ``full``, ``partial`` and ``combined`` summaries (each with
        a ``design`` column) and the two per-gap tables.
    """
    long_lengths = long_lengths or cfg.long_lengths
    pg_f, pts_f = run_gap_experiment(frame, methods, cfg.short_lengths, cfg.n_gaps,
                                     cfg.repeats_short, region, seed, min_observed=1.0,
                                     keep_points=True)
    pg_p, pts_p = run_gap_experiment(frame, methods, long_lengths, cfg.n_gaps,
                                     repeats_long or cfg.repeats_long, region, seed + 1,
                                     min_observed=0.8, keep_points=True)
    full = summarise(pg_f, pts_f).assign(design="fully observed")
    partial = summarise(pg_p, pts_p).assign(design="partially observed")
    short_max = max(cfg.short_lengths)
    combined = pd.concat([full, partial[partial["gap_len"] > short_max]], ignore_index=True)
    return {"full": full, "partial": partial, "combined": combined,
            "per_gap_full": pg_f, "per_gap_partial": pg_p}


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
# Stage 2: tuning and method selection (development period only)
# ----------------------------------------------------------------------------
def tune(cfg: Config, ctx: Context) -> dict:
    """Tune the harmonic model and apply the selection rules on development data.

    Sweeps use the partially observed design at 3, 6 and 12 h, where the
    harmonic settings matter, and ``min_len=6`` so real single dropouts are
    not refitted each time (this does not affect the scored gaps).
    """
    frame = ctx.frames[cfg.primary]
    dev, _ = _regions(len(frame), cfg)
    sweep_lengths = (18, 36, 72)

    def sweep(param, values, base):
        tabs = []
        for v in values:
            kw = {**base, param: v}
            pg = run_gap_experiment(frame, {v: (lambda f, kw=kw: M.fill_harmonic(f, min_len=6, **kw))},
                                    sweep_lengths, cfg.n_gaps, cfg.tune_repeats, dev, cfg.seed)
            tabs.append(summarise(pg).assign(**{param: v}))
        tab = pd.concat(tabs, ignore_index=True)
        best = tab.groupby(param)["MAE"].mean().idxmin()
        return tab, best

    k_tab, k = sweep("k_daily", (0, 1, 2, 3, 4, 6), {"k_weekly": 2, "window_days": 7.0})
    j_tab, j = sweep("k_weekly", (0, 1, 2, 3), {"k_daily": int(k), "window_days": 7.0})
    w_tab, w = sweep("window_days", (2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 21.0),
                     {"k_daily": int(k), "k_weekly": int(j)})
    hk = {"k_daily": int(k), "k_weekly": int(j), "window_days": float(w)}
    for name, tab in [("tune_k_daily", k_tab), ("tune_k_weekly", j_tab), ("tune_window", w_tab)]:
        _save_table(tab.drop(columns="method"), cfg, name)
    plots.sensitivity(k_tab, "k_daily", cfg.figures / "05_sens_k_daily.png",
                      "Daily harmonics (development period)", "Number of daily harmonics K")
    plots.sensitivity(w_tab, "window_days", cfg.figures / "06_sens_window.png",
                      "Fitting window (development period)", "Window half-width (days)")

    # Method comparison on development data, then the switch rule.
    singles = evaluate(frame, method_set(hk), dev, cfg, cfg.seed + 10)
    switch = choose_switch(singles["combined"], "linear", "harmonic", cfg.switch_tolerance)
    # Same seeds -> same gap placements, so the hybrid is a paired comparison.
    hyb = evaluate(frame, {"hybrid": lambda f: M.fill_hybrid(f, switch_len=switch, **hk)},
                   dev, cfg, cfg.seed + 10)
    dev_comb = pd.concat([singles["combined"], hyb["combined"]], ignore_index=True)
    _save_table(dev_comb, cfg, "dev_method_comparison")

    adopted, best_single = adopt_hybrid(dev_comb, "hybrid", cfg.resolution)
    if adopted:
        final = "hybrid"
    else:
        final = simplest_within(singles["combined"], M.COMPLEXITY_ORDER, cfg.switch_tolerance)

    # Optimisation rule (stage 7): if the final method is the harmonic model,
    # deploy it with short-gap skipping when that keeps accuracy within the
    # switch tolerance at every length.
    piv = dev_comb.pivot_table(index="gap_len", columns="method", values="MAE")
    hybrid_ok = bool((piv["hybrid"] <= (1 + cfg.switch_tolerance) * piv["harmonic"]).all())
    deployed = "hybrid" if final in ("harmonic", "hybrid") and hybrid_ok else final

    score = dev_comb.groupby("method")["MAE"].mean().rename("mean_dev_MAE").reset_index()
    params = {**hk, "switch_len": switch, "best_single": best_single,
              "hybrid_adopted_on_accuracy": adopted, "final_method": final,
              "hybrid_accuracy_matches_harmonic": hybrid_ok, "deployed": deployed,
              "dev_scores": score.round(4).to_dict("records")}
    _params_path(cfg).write_text(json.dumps(params, indent=2))
    return params


# ----------------------------------------------------------------------------
# Stage 3: held-out test and independent sensor
# ----------------------------------------------------------------------------
def validate(cfg: Config, ctx: Context, params: dict) -> dict:
    """Final comparison on the held-out test period and on the second sensor."""
    frame = ctx.frames[cfg.primary]
    _, test = _regions(len(frame), cfg)
    hk = _harmonic_kw(params)
    methods = method_set(hk, params["switch_len"])

    res = evaluate(frame, methods, test, cfg, cfg.seed + 20)
    comb = _save_table(res["combined"], cfg, "test_results_primary")
    _save_table(res["full"], cfg, "test_full_observed_primary")
    _save_table(res["partial"], cfg, "test_partial_observed_primary")
    _save_table(res["per_gap_partial"], cfg, "test_per_gap_partial_primary")

    # Agreement of the two designs where both were run.
    both = sorted(set(cfg.short_lengths) & set(cfg.long_lengths))
    agree = (res["full"][res["full"]["gap_len"].isin(both)]
             .merge(res["partial"][res["partial"]["gap_len"].isin(both)],
                    on=["method", "gap_len", "gap_hours"], suffixes=("_full", "_partial")))
    agree["CIs_overlap"] = ((agree["MAE_lo_full"] <= agree["MAE_hi_partial"])
                            & (agree["MAE_lo_partial"] <= agree["MAE_hi_full"]))
    agree = _save_table(agree[["method", "gap_hours", "MAE_full", "MAE_lo_full", "MAE_hi_full",
                               "MAE_partial", "MAE_lo_partial", "MAE_hi_partial", "CIs_overlap"]],
                        cfg, "design_agreement_primary")

    horizon = usable_horizon(comb, cfg.acceptance_mae)
    p95 = comb.pivot_table(index="gap_hours", columns="method", values="P95AE")
    _save_table(horizon.rename_axis("method").reset_index(), cfg, "usable_horizon_primary")

    plots.error_vs_gap(comb, cfg.figures / "03_error_vs_gap_test.png",
                       f"Held-out test period, sensor {cfg.primary}", cfg.acceptance_mae,
                       boundary=max(cfg.short_lengths) / 6)
    plots.error_vs_gap(comb, cfg.figures / "03b_p95_vs_gap_test.png",
                       f"95th-percentile error, sensor {cfg.primary}", cfg.p95_limit,
                       metric="P95AE", boundary=max(cfg.short_lengths) / 6)
    plots.design_agreement(agree, cfg.figures / "03c_design_agreement.png")
    plots.error_by_hour(res["per_gap_partial"], frame, 36, ["linear", "pchip", "harmonic"],
                        cfg.figures / "07_error_by_start_hour.png")

    # Example 12-hour gap from the test period.
    rng = np.random.default_rng(cfg.seed)
    obs = frame["temperature"].notna().to_numpy()
    start = sample_gaps(obs, 72, 1, rng, test)[0]
    masked, _ = apply_gaps(frame, [start], 72)
    fills = {m: methods[m](masked) for m in ("linear", "cubic_spline", "yesterday", "harmonic")}
    plots.example_gap(frame, start, 72, fills, cfg.figures / "04_example_gap.png", context=216)

    # Secondary sensor: never used for decisions; also the long-gap case study.
    f2 = ctx.frames[cfg.secondary]
    res2 = evaluate(f2, methods, None, cfg, cfg.seed + 30, long_lengths=cfg.case_lengths,
                    repeats_long=max(2, cfg.repeats_long // 2))
    comb2 = _save_table(res2["combined"], cfg, "test_results_secondary")
    horizon2 = usable_horizon(comb2, cfg.acceptance_mae)
    _save_table(horizon2.rename_axis("method").reset_index(), cfg, "usable_horizon_secondary")
    plots.error_vs_gap(comb2, cfg.figures / "08_error_vs_gap_secondary.png",
                       f"Independent sensor {cfg.secondary} (unseen room and year)",
                       cfg.acceptance_mae, boundary=max(cfg.short_lengths) / 6)
    return {"primary": comb, "secondary": comb2, "agreement": agree,
            "horizon": horizon, "horizon2": horizon2, "p95": p95}


# ----------------------------------------------------------------------------
# Stage 4: performance
# ----------------------------------------------------------------------------
def performance(cfg: Config, ctx: Context, params: dict) -> dict:
    """Runtime, memory, optimisation steps, scaling, solver stability, profiling."""
    frame = ctx.frames[cfg.primary]
    hk = _harmonic_kw(params)
    methods = method_set(hk, params["switch_len"])
    reps = 1 if cfg.quick else 3

    costs = P.method_costs(frame, methods, repeats=reps)
    flops = P.flop_estimates(frame, switch_len=params["switch_len"], **hk)
    costs = costs.merge(flops, on="method", how="left")
    costs["MFLOP_per_s"] = (costs["flops"] / costs["seconds"] / 1e6).where(costs["flops"] > 0)
    _save_table(costs, cfg, "perf_method_costs")

    # Optimisation of the harmonic method, measured step by step. The
    # secondary sensor is used as the workload too: it has many more and
    # longer real gaps, so the differences are larger.
    steps = {
        "1 naive (basis rebuilt per gap)": lambda f: M.fill_harmonic_naive(f, **hk),
        "2 basis built once": lambda f: M.fill_harmonic(f, **hk),
        "3 + skip short gaps (hybrid)": lambda f: M.fill_hybrid(f, switch_len=params["switch_len"], **hk),
    }
    opt = pd.concat([P.optimisation_steps(ctx.frames[sid], steps, "1 naive (basis rebuilt per gap)",
                                          repeats=reps).assign(sensor=sid)
                     for sid in (cfg.primary, cfg.secondary)], ignore_index=True)
    _save_table(opt, cfg, "perf_optimisation_steps")

    big = P.synthetic_frame(100_000 if cfg.quick else 1_000_000)
    vec = P.vectorisation_speedup(big, M.fill_linear_loop, M.fill_linear, repeats=reps)
    solver = _save_table(P.solver_comparison(frame, **hk), cfg, "perf_solver_comparison")

    sizes = (4_000, 16_000, 64_000) if cfg.quick else (4_000, 16_000, 64_000, 256_000, 1_024_000)
    scale_methods = {k: methods[k] for k in ("linear", "yesterday", "cubic_spline", "pchip",
                                             "hybrid", "harmonic")}
    times, exps = P.scaling(scale_methods, sizes, repeats=reps,
                            max_n={"harmonic": 64_000 if cfg.quick else 256_000})
    _save_table(times, cfg, "perf_scaling")
    _save_table(exps, cfg, "perf_scaling_exponents")
    plots.scaling_plot(times, exps, cfg.figures / "09_scaling.png")

    gap_tabs, slopes = [], {}
    for name in ("harmonic", "hybrid"):
        t, slope = P.gap_count_scaling(methods[name], repeats=reps)
        gap_tabs.append(t.assign(method=name))
        slopes[name] = slope
    gap_tab = _save_table(pd.concat(gap_tabs, ignore_index=True), cfg, "perf_gap_count_scaling")
    plots.gap_count_plot(gap_tab, slopes, cfg.figures / "09b_gap_count_scaling.png")

    prof = "\n".join([
        "=== harmonic (all gaps), primary sensor ===", P.profile_call(methods["harmonic"], frame),
        "=== hybrid, primary sensor ===", P.profile_call(methods["hybrid"], frame)])
    (cfg.root / "results" / "profile_harmonic.txt").write_text(prof)
    misc = {"vectorisation": vec, "gap_count_slopes": slopes}
    (cfg.root / "results" / "perf_misc.json").write_text(json.dumps(misc, indent=2))
    return {"costs": costs, "solver": solver, "exponents": exps, "optimisation": opt,
            "gap_count": gap_tab, **misc}


# ----------------------------------------------------------------------------
# Stage 5: reconstruct real gaps
# ----------------------------------------------------------------------------
def reconstruct(cfg: Config, ctx: Context, params: dict, summaries: dict | None = None) -> dict:
    """Fill the real gaps of both sensors with the deployed method.

    Output columns: ``time``, ``temperature`` (measured or reconstructed),
    ``reconstructed`` (bool), ``gap_len_slots`` (0 if measured) and
    ``expected_abs_error`` - the validated MAE of the selected method for
    that gap length on that sensor, interpolated on log gap length (NaN
    beyond the longest tested length, where the fill is unvalidated).
    """
    hk = _harmonic_kw(params)
    name = params["deployed"]
    method = method_set(hk, params["switch_len"])[name]
    if summaries is None:
        summaries = {}
        for sid, fname in ((cfg.primary, "test_results_primary"),
                           (cfg.secondary, "test_results_secondary")):
            path = cfg.tables / f"{fname}.csv"
            if path.is_file():
                summaries[sid] = pd.read_csv(path)

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
        if sid in summaries:
            g = summaries[sid]
            g = g[g["method"] == name].sort_values("gap_len")
            inside = missing & (gap_len <= g["gap_len"].max())
            expected[inside] = np.interp(np.log(gap_len[inside]), np.log(g["gap_len"]), g["MAE"])
        result = pd.DataFrame({"time": frame["time"], "temperature": filled,
                               "reconstructed": missing, "gap_len_slots": gap_len,
                               "expected_abs_error": expected})
        result.to_csv(cfg.root / "results" / f"reconstructed_{sid}.csv", index=False)
        out[sid] = {"filled_slots": int(missing.sum()), "gaps": int(len(lengths)),
                    "longest_gap_h": round(float(lengths.max()) / 6, 1),
                    "slots_beyond_validated_range": int(np.isnan(expected).sum())}
        if lengths.max() >= 144:
            s = starts[np.argmax(lengths)]
            plots.reconstruction(frame, filled, s, int(lengths.max()),
                                 cfg.figures / f"10_reconstruction_{sid}.png",
                                 plots.LABELS.get(name, name))
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


def _pivot(summary: pd.DataFrame, value: str = "MAE") -> pd.DataFrame:
    piv = summary.pivot_table(index="gap_hours", columns="method", values=value)
    return piv[[m for m in plots.METHOD_ORDER if m in piv.columns]].reset_index()


def write_summary(cfg: Config, ctx: Context, params: dict, val: dict, perf: dict,
                  recon: dict) -> Path:
    """Write ``results/RESULTS.md`` with the headline numbers of a run."""
    sp, ss = ctx.stats[cfg.primary], ctx.stats[cfg.secondary]
    hz = pd.DataFrame({"primary_0869_h": val["horizon"], "secondary_0326_h": val["horizon2"]})
    hz = hz.loc[[m for m in plots.METHOD_ORDER if m in hz.index]].reset_index(names="method")
    costs = perf["costs"][["method", "seconds", "peak_MB", "flops", "MFLOP_per_s", "gaps_fitted"]]
    vec = perf["vectorisation"]
    text = f"""# Results summary (auto-generated)

Run: repeats short/long = {cfg.repeats_short}/{cfg.repeats_long}, gaps per length = {cfg.n_gaps},
seed = {cfg.seed}, quick = {cfg.quick}.

## Data
- CSV rows: {ctx.stats['load']['csv_rows']}.
- Primary {cfg.primary} ({ctx.locations.get(cfg.primary, 'location unknown')}): {sp['observed']} readings,
  {sp['slots']} slots, {sp['missing']} missing ({sp['missing'] / sp['slots']:.1%}), {sp['start']} to {sp['end']}.
- Secondary {cfg.secondary} ({ctx.locations.get(cfg.secondary, 'location unknown')}): {ss['observed']} readings,
  {ss['slots']} slots, {ss['missing']} missing ({ss['missing'] / ss['slots']:.1%}).

## Decisions made on the development period (first {cfg.dev_end:.0%} of primary)
- Harmonic settings: K = {params['k_daily']} daily, J = {params['k_weekly']} weekly harmonics,
  window = {params['window_days']} days.
- Hybrid switch: linear up to {params['switch_len']} slots ({params['switch_len'] / 6:g} h), harmonic above.
- Best single method: {params['best_single']}. Hybrid adopted on accuracy (> {cfg.resolution} °C gain
  somewhere, no loss > {cfg.resolution} °C anywhere): **{params['hybrid_adopted_on_accuracy']}**.
- Final method by the selection rules: **{params['final_method']}**. Hybrid accuracy within
  {cfg.switch_tolerance:.0%} of harmonic at every length: {params['hybrid_accuracy_matches_harmonic']}.
  Deployed implementation: **{params['deployed']}**.

{_md(pd.DataFrame(params['dev_scores']))}

## Held-out test MAE (°C), primary sensor
Up to 2 h: fully observed gaps; above 2 h: partially observed gaps.

{_md(_pivot(val['primary']))}

95th-percentile absolute error (°C), limit {cfg.p95_limit} °C:

{_md(_pivot(val['primary'], 'P95AE'))}

## Usable horizon (longest gap with upper 95% CI of MAE <= {cfg.acceptance_mae} °C)
{_md(hz, "{:g}")}

## Agreement of the two gap designs (primary, test period)
{_md(val['agreement'])}

## Independent sensor and long-gap case study: MAE (°C), sensor {cfg.secondary}
{_md(_pivot(val['secondary']))}

## Performance (filling all real gaps of the primary sensor)
{_md(costs, "{:.4g}")}

Optimisation of the harmonic method (accuracy check: max difference from the naive version):

{_md(perf['optimisation'], "{:.4g}")}

Runtime vs number of gaps at fixed length (log-log slope): harmonic {perf['gap_count_slopes']['harmonic']:.2f},
hybrid {perf['gap_count_slopes']['hybrid']:.2f}.

Loop vs vectorised linear interpolation on 1e6 slots: {vec['loop_s']:.3f} s vs {vec['vectorised_s']:.4f} s
(speed-up {vec['speedup']:.0f}x, identical output: {vec['identical_output']}).

Scaling exponents (slope of log time vs log n):

{_md(perf['exponents'])}

Least-squares solver comparison:

{_md(perf['solver'], "{:.3g}")}

## Reconstruction of real gaps (deployed method: {params['deployed']})
{_md(pd.DataFrame([{'sensor': k, **v} for k, v in recon.items()]))}
"""
    path = cfg.root / "results" / "RESULTS.md"
    path.write_text(text, encoding="utf-8")
    return path
