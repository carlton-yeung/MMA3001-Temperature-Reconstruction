"""Validation by synthetic masking.

Real gaps have no ground truth, so methods are validated on *artificial*
gaps cut out of fully observed stretches: the removed readings are known, so
the reconstruction error is measured exactly.

Design choices (and why):

* Both anchor slots (one before, one after the gap) must hold real
  readings, so every gap has exactly the requested length and every method
  sees the same information.
* Inside the gap at least ``min_observed`` (default 80%) of slots must hold
  real readings. The real sensors drop single readings so often (about 1
  in 14 slots for sensor 326) that no fully observed stretch longer than
  about 5 hours exists; requiring 100% would make long gaps untestable.
  Errors are scored only at slots where the true reading is known.
* Gaps in one repeat never overlap and are separated by a margin, so each
  gap is an independent test case.
* Gaps lie inside a chosen ``region``: the development period for tuning and
  the held-out test period for final results. This keeps model selection
  from seeing the test data.
* Results are pooled over many gaps and repeats, with bootstrap confidence
  intervals over gaps, so small differences between methods are not
  over-interpreted.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def sample_gaps(observed: np.ndarray, gap_len: int, n_gaps: int,
                rng: np.random.Generator, region: tuple[int, int] | None = None,
                margin: int | None = None, min_observed: float = 0.8) -> np.ndarray:
    """Choose random, non-overlapping start slots for synthetic gaps.

    Args:
        observed: Boolean array, ``True`` where a real reading exists.
        gap_len: Length of each gap in slots (>= 1).
        n_gaps: Maximum number of gaps to place.
        rng: NumPy random generator (for reproducibility).
        region: ``(start, end)`` slots the gaps and their anchors must lie in.
        margin: Minimum clear slots between gaps (default ``max(gap_len, 6)``).
        min_observed: Minimum fraction of gap slots holding real readings
            (there must be truth to score against).

    Returns:
        Sorted array of start slots (fewer than ``n_gaps`` if the region
        cannot fit that many).
    """
    if gap_len < 1:
        raise ValueError("gap_len must be >= 1")
    n = len(observed)
    lo, hi = region if region is not None else (0, n)
    margin = max(gap_len, 6) if margin is None else margin
    need = max(1, int(np.ceil(min_observed * gap_len)))

    # Valid start s: anchors s-1 and s+gap_len observed, and enough observed
    # slots inside. A cumulative sum makes the count O(1) per s, O(n) overall.
    csum = np.concatenate([[0], np.cumsum(observed.astype(np.int64))])
    s = np.arange(max(lo, 1), hi - gap_len)  # s-1 >= lo and s+gap_len <= hi-1
    inside = csum[s + gap_len] - csum[s]
    valid = s[observed[s - 1] & observed[s + gap_len] & (inside >= need)]

    chosen, blocked = [], np.zeros(n + 1, dtype=bool)
    for start in rng.permutation(valid):
        if len(chosen) == n_gaps:
            break
        a, b = max(0, start - margin), min(n, start + gap_len + margin)
        if not blocked[a:b].any():
            chosen.append(start)
            blocked[start:start + gap_len] = True
    return np.sort(np.array(chosen, dtype=np.int64))


def apply_gaps(frame: pd.DataFrame, starts: np.ndarray, gap_len: int):
    """Return a copy of ``frame`` with synthetic gaps set to NaN.

    Returns:
        ``(masked_frame, mask)`` where ``mask`` is ``True`` on synthetic-gap
        slots that held a real reading (i.e. where an error can be scored).
    """
    in_gap = np.zeros(len(frame), dtype=bool)
    for s in starts:
        in_gap[s:s + gap_len] = True
    masked = frame.copy()
    masked.loc[in_gap, "temperature"] = np.nan
    return masked, in_gap & frame["temperature"].notna().to_numpy()


def error_metrics(errors: np.ndarray) -> dict:
    """Summary statistics of reconstruction errors (``filled - truth``, °C).

    Returns:
        Dict with ``MAE``, ``RMSE``, ``MaxAE``, ``P95AE`` (95th percentile
        absolute error), ``bias`` (mean signed error) and ``n`` (points).
    """
    e = np.asarray(errors, dtype=float)
    a = np.abs(e)
    return {"MAE": a.mean(), "RMSE": np.sqrt((e ** 2).mean()), "MaxAE": a.max(),
            "P95AE": np.percentile(a, 95), "bias": e.mean(), "n": len(e)}


def run_gap_experiment(frame: pd.DataFrame, methods: dict, lengths, n_gaps: int = 20,
                       repeats: int = 10, region: tuple[int, int] | None = None,
                       seed: int = 0, min_observed: float = 0.8,
                       keep_points: bool = False):
    """Evaluate methods on synthetic gaps of several lengths.

    For each repeat and gap length a fresh set of gaps is sampled; every
    method fills the *same* masked series, so comparisons are paired.

    Args:
        frame: Series providing ground truth.
        methods: Dict of name to callable ``f(frame) -> filled array``.
        lengths: Gap lengths to test (slots of 10 minutes).
        n_gaps: Gaps per length per repeat.
        repeats: Number of independent repeats.
        region: Slots in which gaps may be placed.
        seed: Random seed.
        min_observed: Passed to ``sample_gaps``. ``1.0`` gives the *fully
            observed* design (every slot in the gap has a true value);
            ``0.8`` the *partially observed* design.
        keep_points: Also return every absolute error (for percentiles).

    Returns:
        ``per_gap`` - one row per (method, gap length, repeat, gap) with that
        gap's ``sum_abs``, ``sum_sq``, ``max_abs``, ``sum_err``, ``n`` and the
        method's fill time for that repeat (``seconds``). If ``keep_points``,
        returns ``(per_gap, points)`` where ``points`` maps
        ``(method, gap_len)`` to an array of absolute errors.
    """
    rng = np.random.default_rng(seed)
    truth = frame["temperature"].to_numpy()
    observed = np.isfinite(truth)
    rows, points = [], {}
    for rep in range(repeats):
        for L in lengths:
            starts = sample_gaps(observed, L, n_gaps, rng, region, min_observed=min_observed)
            if len(starts) == 0:
                continue
            masked, _ = apply_gaps(frame, starts, L)
            for name, method in methods.items():
                t0 = time.perf_counter()
                filled = method(masked)
                seconds = time.perf_counter() - t0
                for g, s in enumerate(starts):
                    known = observed[s:s + L]  # score only where truth exists
                    e = filled[s:s + L][known] - truth[s:s + L][known]
                    rows.append((name, L, rep, g, int(s), np.abs(e).sum(), (e ** 2).sum(),
                                 np.abs(e).max(), e.sum(), len(e), seconds))
                    if keep_points:
                        points.setdefault((name, L), []).append(np.abs(e))
    per_gap = pd.DataFrame(rows, columns=["method", "gap_len", "repeat", "gap", "start",
                                          "sum_abs", "sum_sq", "max_abs", "sum_err", "n",
                                          "seconds"])
    if keep_points:
        return per_gap, {k: np.concatenate(v) for k, v in points.items()}
    return per_gap


def summarise(per_gap: pd.DataFrame, points: dict | None = None, n_boot: int = 1000,
              seed: int = 0) -> pd.DataFrame:
    """Pool per-gap results into metrics per (method, gap length).

    MAE confidence intervals are 95% percentile bootstrap intervals over
    gaps (resampling whole gaps, since points within a gap are correlated).

    Args:
        per_gap: Output of ``run_gap_experiment``.
        points: Optional absolute errors from ``run_gap_experiment`` with
            ``keep_points=True``; adds the ``P95AE`` column.
        n_boot: Bootstrap resamples.
        seed: Random seed for the bootstrap.

    Returns:
        DataFrame with ``MAE``, ``MAE_lo``, ``MAE_hi``, ``RMSE``, ``MaxAE``,
        ``P95AE`` (if available), ``bias``, ``gaps``, ``points`` and mean
        fill time ``seconds``.
    """
    rng = np.random.default_rng(seed)
    out = []
    for (method, L), g in per_gap.groupby(["method", "gap_len"], sort=False):
        n = g["n"].to_numpy()
        gap_mae = g["sum_abs"].to_numpy() / n
        boot = rng.choice(gap_mae, size=(n_boot, len(gap_mae))).mean(axis=1)
        row = {
            "method": method, "gap_len": L, "gap_hours": L / 6,
            "MAE": g["sum_abs"].sum() / n.sum(),
            "MAE_lo": np.percentile(boot, 2.5), "MAE_hi": np.percentile(boot, 97.5),
            "RMSE": np.sqrt(g["sum_sq"].sum() / n.sum()),
            "MaxAE": g["max_abs"].max(),
            "bias": g["sum_err"].sum() / n.sum(),
            "gaps": len(g), "points": int(n.sum()),
            "seconds": g.groupby("repeat")["seconds"].first().mean(),
        }
        if points is not None and (method, L) in points:
            row["P95AE"] = float(np.percentile(points[(method, L)], 95))
        out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Decision rules (fixed before seeing test results; see README section 5)
# ----------------------------------------------------------------------------
def choose_switch(summary: pd.DataFrame, short: str = "linear", long: str = "harmonic",
                  tolerance: float = 0.02) -> int:
    """Choose the hybrid switch length from development results.

    Rule: the largest tested gap length ``X`` such that ``short`` is within
    ``tolerance`` (relative) of ``long`` at *every* tested length up to and
    including ``X``. Checking each length separately prevents a real loss
    at one length from being hidden by averaging.

    Args:
        summary: ``summarise`` output containing both methods.
        short, long: Method used below / above the switch.
        tolerance: Allowed relative excess MAE of ``short`` over ``long``.

    Returns:
        Switch length in slots (0 if ``short`` already loses at the
        shortest length).
    """
    piv = summary.pivot_table(index="gap_len", columns="method", values="MAE").sort_index()
    ok = piv[short] <= (1 + tolerance) * piv[long]
    switch = 0
    for L, good in ok.items():
        if not good:
            break
        switch = int(L)
    return switch


def adopt_hybrid(summary: pd.DataFrame, hybrid: str = "hybrid",
                 margin: float = 0.08) -> tuple[bool, str]:
    """Decide whether the hybrid earns its extra complexity.

    Rule: compare the hybrid with the best *single* method (lowest mean MAE
    over the tested lengths). Adopt the hybrid only if it is better by more
    than ``margin`` (the sensor resolution, 0.08 °C) at one or more lengths
    and is not worse by more than ``margin`` at any length.

    Returns:
        ``(adopted, best_single_method)``.
    """
    piv = summary.pivot_table(index="gap_len", columns="method", values="MAE")
    singles = [c for c in piv.columns if c != hybrid]
    best = piv[singles].mean().idxmin()
    diff = piv[best] - piv[hybrid]          # positive = hybrid better
    adopted = bool((diff > margin).any() and (diff >= -margin).all())
    return adopted, str(best)


def simplest_within(summary: pd.DataFrame, order: list, tolerance: float = 0.02) -> str:
    """Pick the simplest method whose mean MAE is within ``tolerance`` of the best.

    Args:
        summary: ``summarise`` output.
        order: Method names from simplest to most complex.
        tolerance: Allowed relative excess over the best mean MAE.
    """
    mean = summary.groupby("method")["MAE"].mean()
    ok = mean[mean <= (1 + tolerance) * mean.min()].index
    return next(m for m in order if m in ok)


def usable_horizon(summary: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """Longest gap (hours) each method fills acceptably, judged conservatively.

    A gap length passes if the *upper* 95% confidence limit of the MAE is at
    or below ``threshold``. The horizon is the longest length such that it
    and every shorter tested length pass (0 if the shortest fails).

    Returns:
        Series of horizon in hours, indexed by method.
    """
    out = {}
    for m, g in summary.groupby("method"):
        h = 0.0
        for _, r in g.sort_values("gap_len").iterrows():
            if r["MAE_hi"] > threshold:
                break
            h = r["gap_hours"]
        out[m] = h
    return pd.Series(out, name="usable_horizon_h")
