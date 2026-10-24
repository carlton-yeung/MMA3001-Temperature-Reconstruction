"""Gap-filling methods for a regularly sampled temperature series.

Every method takes a *frame*: a DataFrame with one row per 10-minute slot and
columns ``time`` (tz-aware local time) and ``temperature`` (°C, NaN where
missing). It returns a NumPy array of the same length in which observed
values are unchanged and every NaN has been filled.

The methods form a ladder from simple to more informed:

==================  =========================================================
``ffill``           Baseline: repeat the last observed value.
``linear``          Baseline: straight line between the readings either side.
``cubic_spline``    Global C2 cubic spline through all observed readings.
``pchip``           Shape-preserving piecewise cubic (no overshoot).
``harmonic``        Local least-squares fit of daily + weekly Fourier terms,
                    with the residual bridged linearly so the fill meets the
                    readings at both ends of the gap.
``boosted``         Gradient-boosted trees that learn a correction to the
                    linear fill from gap context (``BoostedFill``).
``hybrid``          Linear for short gaps, harmonic for long gaps.
==================  =========================================================

Points before the first or after the last observation (edge gaps) are
filled by holding the nearest observed value, for every method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator
from sklearn.ensemble import HistGradientBoostingRegressor

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
# Baselines and classical interpolation
# ----------------------------------------------------------------------------
def fill_ffill(frame: pd.DataFrame) -> np.ndarray:
    """Forward fill: each missing slot takes the last observed value.

    Cost: O(n), no arithmetic. Represents "do nothing clever", the behaviour
    of many building dashboards.
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
    """Pure-Python version of ``fill_linear`` used only as a performance baseline.

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
# Harmonic (Fourier) regression with residual bridging
# ----------------------------------------------------------------------------
def harmonic_design(time: pd.Series, k_daily: int = 3, k_weekly: int = 2) -> np.ndarray:
    """Build the least-squares design matrix for daily and weekly cycles.

    Columns: ``1``, ``trend`` (days), then ``cos/sin(2*pi*k*h/24)`` for
    ``k = 1..k_daily`` and ``cos/sin(2*pi*j*w/168)`` for ``j = 1..k_weekly``,
    where ``h`` is local hour of day and ``w`` is hour of the week. Local
    clock time is used because building schedules follow it (including
    daylight saving).

    Args:
        time: tz-aware timestamps, one per slot.
        k_daily: Number of daily harmonics.
        k_weekly: Number of weekly harmonics.

    Returns:
        Array of shape ``(n, 2 + 2*k_daily + 2*k_weekly)``.
    """
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
            equations ``A^T A c = A^T b`` via Cholesky; about mp^2 + p^3/3
            flops, faster, but squares the condition number of ``A``).

    Returns:
        Coefficient vector of length ``p``.
    """
    if solver == "normal":
        return np.linalg.solve(A.T @ A, A.T @ b)
    if solver == "lstsq":
        return np.linalg.lstsq(A, b, rcond=None)[0]
    raise ValueError(f"Unknown solver {solver!r}")


def fill_harmonic(frame: pd.DataFrame, k_daily: int = 3, k_weekly: int = 2,
                  window_days: float = 7.0, solver: str = "lstsq",
                  design: np.ndarray | None = None, min_len: int = 0) -> np.ndarray:
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

    Cost per gap: building and solving an ``m x p`` least-squares problem
    with ``m`` about ``2 * window_days * 144`` and ``p = 2 + 2(k_daily+k_weekly)``.

    Args:
        frame: Series to fill (see module docstring).
        k_daily: Daily harmonics (shape of the working-day cycle).
        k_weekly: Weekly harmonics (weekday/weekend difference).
        window_days: Half-width of the fitting window (days).
        solver: Least-squares solver, see ``solve_least_squares``.
        design: Optional precomputed ``harmonic_design`` (for speed).
        min_len: Gaps of ``min_len`` slots or fewer are left linear without
            fitting (used by ``fill_hybrid``; saves one solve per short gap).

    Returns:
        Filled temperature array (°C).
    """
    y, obs, _ = _split(frame)
    out = fill_linear(frame)  # fallback values and edge handling
    starts, lengths = gap_runs(~obs)
    long_gap = lengths > min_len
    if not long_gap.any():
        return out
    X = harmonic_design(frame["time"], k_daily, k_weekly) if design is None else design
    n, p = X.shape
    w = int(window_days * STEPS_PER_DAY)

    for s, L in zip(starts[long_gap], lengths[long_gap]):
        lo, hi = max(0, s - w), min(n, s + L + w)
        m = obs[lo:hi]
        if m.sum() < 3 * p:
            continue  # keep linear fallback
        Xw = X[lo:hi].copy()
        Xw[:, 1] -= Xw[:, 1].mean()  # centre the trend column for conditioning
        coef = solve_least_squares(Xw[m], y[lo:hi][m], solver)
        pred = Xw @ coef
        resid = y[lo:hi] - pred
        r_left = resid[s - 1 - lo] if s > 0 else np.nan
        r_right = resid[s + L - lo] if s + L < n else np.nan
        if np.isnan(r_left):
            r_left = r_right
        if np.isnan(r_right):
            r_right = r_left
        frac = np.arange(1, L + 1) / (L + 1)
        out[s:s + L] = pred[s - lo:s + L - lo] + (1 - frac) * r_left + frac * r_right
    return out


# ----------------------------------------------------------------------------
# Machine learning: gradient-boosted correction to the linear fill
# ----------------------------------------------------------------------------
FEATURES = ["log_gap_len", "frac", "steps_from_left", "steps_to_right",
            "hour_sin", "hour_cos", "weekend", "left", "right",
            "slope_left", "slope_right", "harmonic_minus_linear"]
"""Names of the features used by ``BoostedFill`` (for reporting)."""


def gap_features(frame: pd.DataFrame, k_daily: int = 3, k_weekly: int = 2,
                 window_days: float = 7.0):
    """Describe every missing slot by its gap context.

    Args:
        frame: Series with gaps.
        k_daily, k_weekly, window_days: Settings for the harmonic feature.

    Returns:
        ``(idx, F, linear, harmonic)`` - indices of missing slots, a feature
        matrix (columns ``FEATURES``; NaN where a neighbour is unavailable,
        which histogram gradient boosting handles natively) and the full
        linear and harmonic fills.
    """
    y, obs, _ = _split(frame)
    linear = fill_linear(frame)
    harmonic = fill_harmonic(frame, k_daily, k_weekly, window_days)
    starts, lengths = gap_runs(~obs)
    gid = np.repeat(np.arange(len(starts)), lengths)
    s, L = starts[gid], lengths[gid]
    idx = s + np.concatenate([np.arange(n) for n in lengths]) if len(s) else s
    n = len(y)

    def value_at(i):
        ok = (i >= 0) & (i < n)
        v = np.full(len(i), np.nan)
        v[ok] = y[i[ok]]
        return v

    left, right = value_at(s - 1), value_at(s + L)
    hour = (frame["time"].dt.hour + frame["time"].dt.minute / 60).to_numpy()[idx]
    F = np.column_stack([
        np.log(L), (idx - s + 1) / (L + 1), idx - s + 1, s + L - idx,
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        frame["time"].dt.dayofweek.to_numpy()[idx] >= 5,
        left, right,
        (left - value_at(s - 7)) / 6, (value_at(s + L + 6) - right) / 6,
        harmonic[idx] - linear[idx],
    ])
    return idx, F, linear, harmonic


class BoostedFill:
    """Gradient-boosted trees that predict the error of a base fill.

    The model learns ``truth - base_fill`` from synthetic gaps cut out of a
    training period, using the features in ``gap_features`` (gap length,
    position in the gap, time of day, weekend flag, edge values and slopes,
    and the harmonic model's opinion). Predicting a *correction* rather than
    the temperature itself means a poorly trained model degrades towards the
    base method instead of producing absurd values.

    Each training gap gets equal total weight (rows weighted by
    ``1 / gap_len``). Without this, a 3-day gap contributes 432 times more
    rows than a 10-minute gap and the model neglects short gaps.

    Args:
        base: ``"linear"`` or ``"harmonic"`` - the fill being corrected.
        max_iter: Boosting iterations (upper bound; early stopping is used).
        learning_rate: Shrinkage per iteration.
        max_leaf_nodes: Tree complexity.
        random_state: Seed for reproducibility.
        k_daily, k_weekly, window_days: Harmonic feature settings.

    Example:
        >>> model = BoostedFill().fit(train_frame, train_slots)   # doctest: +SKIP
        >>> filled = model(frame_with_gaps)                         # doctest: +SKIP
    """

    def __init__(self, base: str = "harmonic", max_iter: int = 400, learning_rate: float = 0.05,
                 max_leaf_nodes: int = 31, random_state: int = 0,
                 k_daily: int = 3, k_weekly: int = 2, window_days: float = 7.0):
        if base not in ("linear", "harmonic"):
            raise ValueError("base must be 'linear' or 'harmonic'")
        self.base = base
        self.harmonic_kw = dict(k_daily=k_daily, k_weekly=k_weekly, window_days=window_days)
        self.random_state = random_state
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
            early_stopping=True, validation_fraction=0.15, random_state=random_state)
        self.n_train_rows_ = 0

    def fit(self, frame: pd.DataFrame, region: tuple[int, int],
            lengths=(1, 3, 6, 18, 36, 72, 144, 288, 432), n_gaps: int = 30,
            repeats: int = 4) -> "BoostedFill":
        """Train on synthetic gaps placed inside ``region`` of ``frame``.

        Args:
            frame: Series providing ground truth.
            region: ``(start_slot, end_slot)`` the training gaps must lie in
                (the training period; never the test period).
            lengths: Gap lengths (slots) to simulate.
            n_gaps: Gaps per length per repeat.
            repeats: Independent masking repeats.

        Returns:
            ``self`` (fitted).
        """
        from .validation import apply_gaps, sample_gaps  # avoid circular import

        rng = np.random.default_rng(self.random_state)
        observed = frame["temperature"].notna().to_numpy()
        truth = frame["temperature"].to_numpy()
        Fs, targets, weights = [], [], []
        for _ in range(repeats):
            for L in lengths:
                starts = sample_gaps(observed, L, n_gaps, rng, region)
                if len(starts) == 0:
                    continue
                masked, mask = apply_gaps(frame, starts, L)
                idx, F, linear, harmonic = gap_features(masked, **self.harmonic_kw)
                base = linear if self.base == "linear" else harmonic
                keep = mask[idx]  # only synthetic-gap slots with a real reading
                Fs.append(F[keep])
                targets.append(truth[idx[keep]] - base[idx[keep]])
                weights.append(np.full(keep.sum(), 1.0 / L))
        X, t = np.vstack(Fs), np.concatenate(targets)
        self.n_train_rows_ = len(t)
        self.model.fit(X, t, sample_weight=np.concatenate(weights))
        return self

    def __call__(self, frame: pd.DataFrame) -> np.ndarray:
        """Fill every gap in ``frame`` using the fitted model."""
        idx, F, linear, harmonic = gap_features(frame, **self.harmonic_kw)
        base = linear if self.base == "linear" else harmonic
        out = base.copy()
        if len(idx):
            out[idx] = base[idx] + self.model.predict(F)
        y, obs, _ = _split(frame)
        return _hold_edges(np.where(obs, y, out), y, obs)


# ----------------------------------------------------------------------------
# Hybrid and registry
# ----------------------------------------------------------------------------
def fill_hybrid(frame: pd.DataFrame, switch_len: int = 6, **harmonic_kw) -> np.ndarray:
    """Linear interpolation for gaps up to ``switch_len`` slots, harmonic above.

    Short gaps are dominated by sensor noise and quantisation (0.1 °C), where
    linear is already near-optimal and cheapest; long gaps need the daily
    cycle that only the harmonic model supplies. ``switch_len`` is chosen on
    the training period (see ``tempfill.experiments``).
    """
    return fill_harmonic(frame, min_len=switch_len, **harmonic_kw)


BASIC_METHODS = {
    "ffill": fill_ffill,
    "linear": fill_linear,
    "cubic_spline": fill_cubic_spline,
    "pchip": fill_pchip,
    "harmonic": fill_harmonic,
}
"""Methods that need no training, keyed by the name used in results tables."""
