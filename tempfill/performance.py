"""Runtime, memory, scaling and numerical-stability measurements.

Tools used: ``time.perf_counter`` for timing (median of repeats to suppress
noise), ``tracemalloc`` for peak Python memory, ``cProfile`` for locating
hot spots, and analytical flop counts to relate measured times to the
arithmetic each method performs.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time
import tracemalloc

import numpy as np
import pandas as pd

from .data import STEP_S, gap_runs
from .methods import STEPS_PER_DAY, harmonic_design, solve_least_squares


def time_call(func, *args, repeats: int = 5, **kwargs) -> float:
    """Median wall-clock time of ``func(*args, **kwargs)`` in seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def peak_memory_mb(func, *args, **kwargs) -> float:
    """Peak memory allocated by Python while running ``func`` (MB).

    ``tracemalloc`` tracks allocations made through Python's allocator,
    including NumPy arrays, so it measures the working memory of a method.
    """
    tracemalloc.start()
    try:
        func(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1e6


def synthetic_frame(n: int, missing_fraction: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Create a realistic test series of ``n`` 10-minute slots.

    Temperature follows a daily cycle with weekday/weekend difference plus
    noise rounded to 0.1 °C (the sensor resolution). Single readings are
    dropped at random with probability ``missing_fraction``, and a few
    multi-hour gaps are added, mimicking the real data.

    Args:
        n: Number of slots.
        missing_fraction: Probability a single slot is missing.
        seed: Random seed.

    Returns:
        Frame with ``time`` and ``temperature`` columns.
    """
    rng = np.random.default_rng(seed)
    time_ = pd.Series(pd.date_range("2025-01-06", periods=n, freq=f"{STEP_S}s",
                                    tz="Australia/Melbourne"))
    hour = time_.dt.hour.to_numpy() + time_.dt.minute.to_numpy() / 60
    weekday = time_.dt.dayofweek.to_numpy() < 5
    temp = (21 + 2.0 * weekday * np.exp(-((hour - 14) / 4) ** 2)
            + 0.8 * np.cos(2 * np.pi * (hour - 15) / 24) + rng.normal(0, 0.15, n))
    temp = np.round(temp, 1)
    temp[rng.random(n) < missing_fraction] = np.nan
    for s in rng.integers(0, max(1, n - 200), size=max(1, n // 5000)):
        temp[s:s + rng.integers(18, 150)] = np.nan
    temp[[0, -1]] = 21.0  # keep both ends observed
    return pd.DataFrame({"time": time_, "temperature": temp})


def method_costs(frame: pd.DataFrame, methods: dict, repeats: int = 5) -> pd.DataFrame:
    """Time and memory for each method filling every real gap in ``frame``.

    Returns:
        DataFrame with ``method``, ``seconds`` (median), ``peak_MB`` and the
        throughput ``us_per_missing`` (microseconds per filled slot).
    """
    n_missing = int(frame["temperature"].isna().sum())
    rows = []
    for name, method in methods.items():
        sec = time_call(method, frame, repeats=repeats)
        rows.append({"method": name, "seconds": sec,
                     "peak_MB": peak_memory_mb(method, frame),
                     "us_per_missing": 1e6 * sec / max(n_missing, 1)})
    return pd.DataFrame(rows)


def scaling(methods: dict, sizes=(4_000, 16_000, 64_000, 256_000, 1_024_000),
            repeats: int = 3, max_n: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure how runtime grows with series length.

    The empirical complexity exponent is the slope of log(time) vs log(n),
    fitted by least squares over the largest three sizes run. An
    exponent near 1 means O(n); a value well below 1 means fixed overheads
    (Python calls, array allocation) still dominate at these sizes.

    Args:
        methods: Dict of name to fill function.
        sizes: Series lengths to test.
        repeats: Timing repeats (median taken).
        max_n: Optional per-method size cap for slow methods.

    Returns:
        ``(times, exponents)`` - long table of (method, n, seconds) and the
        fitted exponent per method (from the largest three sizes run).
    """
    max_n = max_n or {}
    rows = []
    for n in sizes:
        frame = synthetic_frame(n)
        for name, method in methods.items():
            if n <= max_n.get(name, np.inf):
                rows.append({"method": name, "n": n,
                             "seconds": time_call(method, frame, repeats=repeats)})
    times = pd.DataFrame(rows)
    exps = []
    for m, g in times.groupby("method", sort=False):
        g = g.nlargest(3, "n")
        exps.append({"method": m, "exponent": np.polyfit(np.log(g["n"]), np.log(g["seconds"]), 1)[0],
                     "n_range": f"{g['n'].min()}-{g['n'].max()}"})
    return times, pd.DataFrame(exps)


def solver_comparison(frame: pd.DataFrame, k_daily: int = 2, k_weekly: int = 2,
                      window_days: float = 7.0, repeats: int = 50) -> pd.DataFrame:
    """Compare least-squares solvers and the effect of centring the trend.

    Uses one representative fitting window (the start of the series). For
    each variant reports the condition number of the design matrix and of
    the normal-equations matrix, solve time, and the coefficient difference
    from the SVD solution (a measure of numerical error).
    """
    y = frame["temperature"].to_numpy()
    w = int(2 * window_days * STEPS_PER_DAY)
    X = harmonic_design(frame["time"], k_daily, k_weekly)[:w]
    m = np.isfinite(y[:w])
    rows = []
    for centred in (False, True):
        A = X[m].copy()
        if centred:
            A[:, 1] -= A[:, 1].mean()
        else:
            A[:, 1] += 200.0  # a window ~200 days into the record, uncentred
        b = y[:w][m]
        ref = solve_least_squares(A, b, "lstsq")
        for solver in ("lstsq", "normal"):
            c = solve_least_squares(A, b, solver)
            rows.append({
                "trend_centred": centred, "solver": solver,
                "cond_A": np.linalg.cond(A), "cond_AtA": np.linalg.cond(A.T @ A),
                "seconds": time_call(solve_least_squares, A, b, solver, repeats=repeats),
                "max_coef_diff_vs_svd": float(np.abs(c - ref).max()),
                "rows_m": A.shape[0], "params_p": A.shape[1],
            })
    return pd.DataFrame(rows)


def gap_count_scaling(method, n: int = 64_000,
                      fractions=(0.01, 0.02, 0.05, 0.1, 0.2), repeats: int = 3) -> tuple[pd.DataFrame, float]:
    """Runtime against the number of gaps, at fixed series length.

    Tests the cost model of the harmonic method: one least-squares fit per
    gap, so time should grow in proportion to the number of gaps (slope 1
    on a log-log plot) rather than with series length.

    Args:
        method: Fill function to time.
        n: Fixed series length (slots).
        fractions: Probabilities of a single slot being missing.
        repeats: Timing repeats.

    Returns:
        ``(table, slope)`` - (missing_fraction, gaps, seconds, us_per_gap)
        rows, and the fitted log-log slope of time against gap count.
    """
    rows = []
    for f in fractions:
        frame = synthetic_frame(n, missing_fraction=f)
        n_gaps = len(gap_runs(frame["temperature"].isna().to_numpy())[0])
        sec = time_call(method, frame, repeats=repeats)
        rows.append({"missing_fraction": f, "gaps": n_gaps, "seconds": sec,
                     "us_per_gap": 1e6 * sec / n_gaps})
    t = pd.DataFrame(rows)
    slope = float(np.polyfit(np.log(t["gaps"]), np.log(t["seconds"]), 1)[0])
    return t, slope


def optimisation_steps(frame: pd.DataFrame, steps: dict, reference: str,
                       repeats: int = 3) -> pd.DataFrame:
    """Time successive versions of a method and check they agree.

    Args:
        frame: Series to fill (its real gaps are the workload).
        steps: Ordered dict of label to fill function, from the unoptimised
            reference to the final version.
        reference: Label of the version whose output the others are
            compared against.

    Returns:
        DataFrame with ``version``, ``seconds``, ``speedup`` (vs the first
        version), ``peak_MB`` and ``max_diff_C`` (largest difference from
        the reference output, °C).
    """
    ref = steps[reference](frame)
    rows, t0 = [], None
    for label, func in steps.items():
        sec = time_call(func, frame, repeats=repeats)
        t0 = sec if t0 is None else t0
        rows.append({"version": label, "seconds": sec, "speedup": t0 / sec,
                     "peak_MB": peak_memory_mb(func, frame),
                     "max_diff_C": float(np.abs(func(frame) - ref).max())})
    return pd.DataFrame(rows)


def vectorisation_speedup(frame: pd.DataFrame, loop_func, vec_func, repeats: int = 3) -> dict:
    """Time a pure-Python loop against its vectorised equivalent.

    Returns:
        Dict with both times, the speed-up factor and whether outputs agree.
    """
    t_loop = time_call(loop_func, frame, repeats=repeats)
    t_vec = time_call(vec_func, frame, repeats=repeats)
    same = bool(np.allclose(loop_func(frame), vec_func(frame), equal_nan=True))
    return {"loop_s": t_loop, "vectorised_s": t_vec, "speedup": t_loop / t_vec,
            "identical_output": same}


def profile_call(func, *args, top: int = 12, **kwargs) -> str:
    """Run ``func`` under cProfile; return the top functions by cumulative time."""
    prof = cProfile.Profile()
    prof.enable()
    func(*args, **kwargs)
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(top)
    return buf.getvalue()


def flop_estimates(frame: pd.DataFrame, k_daily: int = 2, k_weekly: int = 2,
                   window_days: float = 7.0, switch_len: int = 6) -> pd.DataFrame:
    """Analytical floating-point operation counts for filling ``frame``'s gaps.

    Counts are leading-order estimates, used to compare methods and to
    judge whether measured times are compute-bound or overhead-bound:

    * linear: ~5 flops per missing slot (weights and blend).
    * cubic spline: banded solve ~8n plus ~10 flops per evaluated slot.
    * PCHIP: ~20 flops per knot for slopes plus ~10 per evaluated slot.
    * harmonic: per fitted gap, an ``m x p`` least-squares solve by SVD
      (~2mp^2 + 11p^3) plus prediction (2mp), with ``m`` the window length.
      Building the design matrix costs ~``n * p`` trig evaluations once.

    Returns:
        DataFrame with ``method``, ``flops`` and ``gaps_fitted``.
    """
    y = frame["temperature"].to_numpy()
    obs = np.isfinite(y)
    n, n_obs, n_miss = len(y), int(obs.sum()), int((~obs).sum())
    p = 2 + 2 * (k_daily + k_weekly)
    _, lengths = gap_runs(~obs)
    m = 2 * window_days * STEPS_PER_DAY + lengths

    def harmonic_flops(sel):
        if not sel.any():
            return 0.0  # no gap is fitted, so the design matrix is never built
        mm = m[sel]
        return float((2 * mm * p**2 + 11 * p**3 + 2 * mm * p).sum() + 20 * n * p)

    return pd.DataFrame([
        {"method": "linear", "flops": 5.0 * n_miss, "gaps_fitted": 0},
        {"method": "cubic_spline", "flops": 8.0 * n_obs + 10.0 * n_miss, "gaps_fitted": 0},
        {"method": "pchip", "flops": 20.0 * n_obs + 10.0 * n_miss, "gaps_fitted": 0},
        {"method": "harmonic", "flops": harmonic_flops(lengths > 0),
         "gaps_fitted": int(len(lengths))},
        {"method": "hybrid", "flops": harmonic_flops(lengths > switch_len),
         "gaps_fitted": int((lengths > switch_len).sum())},
    ])
