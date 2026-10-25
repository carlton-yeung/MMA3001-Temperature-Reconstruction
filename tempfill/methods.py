"""Gap-filling methods for a regularly sampled temperature series.

Every method takes a *frame*: a DataFrame with one row per 10-minute slot and
columns ``time`` (tz-aware local time) and ``temperature`` (°C, NaN where
missing). It returns a NumPy array of the same length in which observed
values are unchanged and every NaN has been filled.

Baselines (what would happen without this project, and the simplest
alternatives):

==================  =========================================================
``ffill``           Repeat the last observed value (what a dashboard does).
``linear``          Straight line between the readings either side.
``yesterday``       Reading from the same time on the previous day (the
                    simplest method that uses the daily pattern).
==================  =========================================================

Methods investigated:

==================  =========================================================
``cubic_spline``    Global C2 cubic spline through all observed readings.
``pchip``           Shape-preserving piecewise cubic (no overshoot).
``harmonic``        Local least-squares fit of daily + weekly Fourier terms,
                    with the residual bridged linearly so the fill meets the
                    readings at both ends of the gap.
``hybrid``          Linear for short gaps, harmonic for long gaps (also the
                    main runtime optimisation of ``harmonic``).
==================  =========================================================

Points before the first or after the last observation (edge gaps) are
filled by holding the nearest observed value, for every method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

from .data import gap_runs

STEPS_PER_DAY = 144
"""Number of 10-minute slots per day."""


def _split(frame: pd.DataFrame):
    """Return ``(y, observed_mask, slot_index)`` for a frame."""
    y = frame["temperature"].to_numpy(dtype=float, copy=True)  # writable copy
    obs = np.isfinite(y)
    if obs.sum() < 2:
        raise ValueError("Need at least two observed readings to fill gaps.")
    return y, obs, np.arange(len(y))


def _hold_edges(out: np.ndarray, y: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fill any NaN left before the first / after the last observation."""
    first, last = np.flatnonzero(obs)[[0, -1]]
    out[:first] = y[first]
    out[last + 1:] = y[last]
    return out


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------
def fill_ffill(frame: pd.DataFrame) -> np.ndarray:
    """Forward fill: each missing slot takes the last observed value.

    Cost: O(n), no arithmetic. The only method usable *during* a live
    outage, since it needs no reading after the gap.
    """
    y, obs, _ = _split(frame)
    return _hold_edges(pd.Series(y).ffill().to_numpy(copy=True), y, obs)


def fill_linear(frame: pd.DataFrame) -> np.ndarray:
    """Linear interpolation between the readings either side of each gap.

    Uses ``numpy.interp`` (vectorised, compiled). Cost: O(n) to locate
    neighbours plus about 5 flops per missing slot. Error is O(h^2) for a
    smooth signal sampled every h.
    """
    y, obs, x = _split(frame)
    out = y.copy()
    out[~obs] = np.interp(x[~obs], x[obs], y[obs])  # interp holds edge values
    return out


def fill_linear_loop(frame: pd.DataFrame) -> np.ndarray:
    """Pure-Python version of ``fill_linear`` used only as a performance reference.

    Gives identical results; exists so the speed-up from vectorisation can
    be measured (see ``tempfill.performance``).
    """
    y, obs, _ = _split(frame)
    out = y.copy()
    n, last = len(y), None
    i = 0
    while i < n:
        if obs[i]:
            last = i
            i += 1
            continue
        j = i
        while j < n and not obs[j]:
            j += 1
        for k in range(i, j):
            if last is None:
                out[k] = y[j]
            elif j == n:
                out[k] = y[last]
            else:
                w = (k - last) / (j - last)
                out[k] = (1 - w) * y[last] + w * y[j]
        i = j
    return out


def fill_yesterday(frame: pd.DataFrame, max_days: int = 7) -> np.ndarray:
    """Fill each missing slot with the reading at the same time on an earlier day.

    Looks back 1 day; if that reading is also missing, 2 days, and so on up
    to ``max_days``. Slots with no earlier reading (e.g. in the first day)
    fall back to linear interpolation.

    This baseline uses a completely different source of information from
    interpolation: the shape of a previous day rather than the readings at
    the gap edges. It is blind to day-to-day level changes (a warmer day
    gives a constant offset and a jump at the gap edges) and to weekday vs
    weekend differences.

    Cost: O(n * max_days) array look-ups, no arithmetic.
    """
    y, obs, _ = _split(frame)
    out = y.copy()
    todo = ~obs
    for k in range(1, max_days + 1):
        if not todo.any():
            break
        lag = k * STEPS_PER_DAY
        src = np.full(len(y), np.nan)
        src[lag:] = y[:-lag]
        take = todo & np.isfinite(src)
        out[take] = src[take]
        todo &= ~take
    if todo.any():
        out[todo] = fill_linear(frame)[todo]
    return out


# ----------------------------------------------------------------------------
# Classical interpolation
# ----------------------------------------------------------------------------
def fill_cubic_spline(frame: pd.DataFrame) -> np.ndarray:
    """Global cubic spline (not-a-knot) through every observed reading.

    The spline coefficients come from one banded (tridiagonal-type) linear
    solve, O(n) in time and memory (about 8n flops). Very accurate for short
    gaps in smooth data (error O(h^4)) but can overshoot inside long gaps
    because nothing constrains the curve between distant knots.
    """
    y, obs, x = _split(frame)
    out = y.copy()
    out[~obs] = CubicSpline(x[obs], y[obs], extrapolate=False)(x[~obs])
    return _hold_edges(out, y, obs)


def fill_pchip(frame: pd.DataFrame) -> np.ndarray:
    """Piecewise cubic Hermite interpolation (Fritsch-Carlson PCHIP).

    Slopes are chosen so the curve never overshoots the neighbouring
    readings; no linear system is solved. O(n). Safer than a spline for long
    gaps, but still cannot invent a daily cycle inside a gap.
    """
    y, obs, x = _split(frame)
    out = y.copy()
    out[~obs] = PchipInterpolator(x[obs], y[obs], extrapolate=False)(x[~obs])
    return _hold_edges(out, y, obs)


# ----------------------------------------------------------------------------
# Harmonic (Fourier) least squares with residual bridging
# ----------------------------------------------------------------------------
def harmonic_design(time: pd.Series, k_daily: int = 2, k_weekly: int = 2) -> np.ndarray:
    """Build the least-squares design matrix for daily and weekly cycles.

    Columns: ``1``, ``trend`` (days since ``time[0]``), then
    ``cos/sin(2*pi*k*h/24)`` for ``k = 1..k_daily`` and
    ``cos/sin(2*pi*j*w/168)`` for ``j = 1..k_weekly``, where ``h`` is local
    hour of day and ``w`` is hour of the week. Local clock time is used
    because building schedules follow it (including daylight saving).

    Args:
        time: tz-aware timestamps, one per slot.
        k_daily: Number of daily harmonics.
        k_weekly: Number of weekly harmonics.

    Returns:
        Array of shape ``(n, 2 + 2*k_daily + 2*k_weekly)``.
    """
    time = pd.Series(time).reset_index(drop=True)
    hour = (time.dt.hour + time.dt.minute / 60 + time.dt.second / 3600).to_numpy()
    week_hour = time.dt.dayofweek.to_numpy() * 24 + hour
    days = (time - time.iloc[0]).dt.total_seconds().to_numpy() / 86400
    cols = [np.ones_like(hour), days]
    for k in range(1, k_daily + 1):
        cols += [np.cos(2 * np.pi * k * hour / 24), np.sin(2 * np.pi * k * hour / 24)]
    for j in range(1, k_weekly + 1):
        cols += [np.cos(2 * np.pi * j * week_hour / 168), np.sin(2 * np.pi * j * week_hour / 168)]
    return np.column_stack(cols)


def solve_least_squares(A: np.ndarray, b: np.ndarray, solver: str = "lstsq") -> np.ndarray:
    """Solve ``min ||A c - b||`` for the coefficients ``c``.

    Args:
        A: Design matrix, shape ``(m, p)``.
        b: Observations, shape ``(m,)``.
        solver: ``"lstsq"`` (SVD-based ``numpy.linalg.lstsq``; numerically
            stable, about 2mp^2 + 11p^3 flops) or ``"normal"`` (normal
            equations ``A^T A c = A^T b``; about mp^2 + p^3/3 flops, faster,
            but squares the condition number of ``A``).

    Returns:
        Coefficient vector of length ``p``.
    """
    if solver == "normal":
        return np.linalg.solve(A.T @ A, A.T @ b)
    if solver == "lstsq":
        return np.linalg.lstsq(A, b, rcond=None)[0]
    raise ValueError(f"Unknown solver {solver!r}")


def _bridge(pred: np.ndarray, resid: np.ndarray, s: int, L: int, lo: int, n: int) -> np.ndarray:
    """Model prediction inside a gap plus the linearly bridged edge residuals."""
    r_left = resid[s - 1 - lo] if s > 0 else np.nan
    r_right = resid[s + L - lo] if s + L < n else np.nan
    if np.isnan(r_left):
        r_left = r_right
    if np.isnan(r_right):
        r_right = r_left
    frac = np.arange(1, L + 1) / (L + 1)
    return pred[s - lo:s + L - lo] + (1 - frac) * r_left + frac * r_right


def fill_harmonic(frame: pd.DataFrame, k_daily: int = 2, k_weekly: int = 2,
                  window_days: float = 7.0, solver: str = "lstsq",
                  min_len: int = 0, centre_trend: bool = True) -> np.ndarray:
    """Fill each gap with a local Fourier model anchored to the gap edges.

    For every gap:

    1. Fit ``T(t) = c . phi(t)`` (constant, trend, daily and weekly
       harmonics; see ``harmonic_design``) by least squares to the observed
       readings within ``window_days`` either side of the gap.
    2. Compute the model residual at the last reading before and the first
       reading after the gap, and interpolate it linearly across the gap.
    3. Fill = model prediction + interpolated residual.

    Step 2 makes the fill continuous with the real data, so the model only
    has to supply the *shape* of the daily cycle, not its absolute level.
    A gap whose window has too few readings (fewer than 3 per parameter)
    falls back to linear interpolation.

    The design matrix is built once for the whole series and sliced per gap
    (compare ``fill_harmonic_naive``).

    Cost per fitted gap: an ``m x p`` least-squares problem with
    ``m`` about ``2 * window_days * 144`` and ``p = 2 + 2(k_daily+k_weekly)``.

    Args:
        frame: Series to fill (see module docstring).
        k_daily: Daily harmonics (shape of the working-day cycle).
        k_weekly: Weekly harmonics (weekday/weekend difference).
        window_days: Half-width of the fitting window (days).
        solver: Least-squares solver, see ``solve_least_squares``.
        min_len: Gaps of ``min_len`` slots or fewer are left linear without
            fitting (used by ``fill_hybrid``; saves one solve per short gap).
        centre_trend: Subtract the window mean from the trend column before
            fitting. Improves conditioning (see ``performance.solver_comparison``).

    Returns:
        Filled temperature array (°C).
    """
    y, obs, _ = _split(frame)
    out = fill_linear(frame)  # fallback values and edge handling
    starts, lengths = gap_runs(~obs)
    long_gap = lengths > min_len
    if not long_gap.any():
        return out
    X = harmonic_design(frame["time"], k_daily, k_weekly)
    n, p = X.shape
    w = int(window_days * STEPS_PER_DAY)

    for s, L in zip(starts[long_gap], lengths[long_gap]):
        lo, hi = max(0, s - w), min(n, s + L + w)
        m = obs[lo:hi]
        if m.sum() < 3 * p:
            continue  # keep linear fallback
        Xw = X[lo:hi].copy()
        if centre_trend:
            Xw[:, 1] -= Xw[:, 1].mean()
        coef = solve_least_squares(Xw[m], y[lo:hi][m], solver)
        pred = Xw @ coef
        out[s:s + L] = _bridge(pred, y[lo:hi] - pred, s, L, lo, n)
    return out


def fill_harmonic_naive(frame: pd.DataFrame, k_daily: int = 2, k_weekly: int = 2,
                        window_days: float = 7.0) -> np.ndarray:
    """Unoptimised reference version of ``fill_harmonic``.

    Rebuilds the design matrix (all the sine and cosine evaluations) from
    the timestamps of every gap's window, and fits every gap including
    single missed readings. Written deliberately as the "before" case for
    the optimisation study; results match ``fill_harmonic`` exactly.
    """
    y, obs, _ = _split(frame)
    out = fill_linear(frame)
    n = len(y)
    w = int(window_days * STEPS_PER_DAY)
    times = frame["time"]
    for s, L in zip(*gap_runs(~obs)):
        lo, hi = max(0, s - w), min(n, s + L + w)
        # Same columns as harmonic_design on the full series: rebuild then
        # shift the trend so it matches the full-series origin.
        Xw = harmonic_design(times.iloc[lo:hi], k_daily, k_weekly)
        m = obs[lo:hi]
        if m.sum() < 3 * Xw.shape[1]:
            continue
        Xw[:, 1] -= Xw[:, 1].mean()
        coef = solve_least_squares(Xw[m], y[lo:hi][m])
        pred = Xw @ coef
        out[s:s + L] = _bridge(pred, y[lo:hi] - pred, s, L, lo, n)
    return out


def fill_hybrid(frame: pd.DataFrame, switch_len: int = 6, **harmonic_kw) -> np.ndarray:
    """Linear interpolation for gaps up to ``switch_len`` slots, harmonic above.

    Short gaps are dominated by sensor noise and the 0.1 °C quantisation,
    where linear is already as good as the harmonic model and needs no
    least-squares fit; long gaps need the daily cycle that only the harmonic
    model supplies. ``switch_len`` is chosen on development data only (see
    ``validation.choose_switch``).
    """
    return fill_harmonic(frame, min_len=switch_len, **harmonic_kw)


BASELINES = {"ffill": fill_ffill, "linear": fill_linear, "yesterday": fill_yesterday}
"""Baseline methods, keyed by the name used in results tables."""

COMPLEXITY_ORDER = ["ffill", "linear", "yesterday", "pchip", "cubic_spline", "harmonic", "hybrid"]
"""Methods from simplest to most complex; used to break ties in favour of simplicity."""
