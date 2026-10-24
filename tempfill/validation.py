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
* Gaps lie inside a chosen ``region``: the training period for tuning and
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
                       seed: int = 0) -> pd.DataFrame:
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

    Returns:
        One row per (method, gap length, repeat, gap) with that gap's
        ``sum_abs``, ``sum_sq``, ``max_abs``, ``sum_err``, ``n`` and the
        method's fill time for that repeat (``seconds``).
    """
    rng = np.random.default_rng(seed)
    truth = frame["temperature"].to_numpy()
    observed = np.isfinite(truth)
    rows = []
    for rep in range(repeats):
        for L in lengths:
            starts = sample_gaps(observed, L, n_gaps, rng, region)
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
    return pd.DataFrame(rows, columns=["method", "gap_len", "repeat", "gap", "start",
                                       "sum_abs", "sum_sq", "max_abs", "sum_err", "n",
                                       "seconds"])


def summarise(per_gap: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Pool per-gap results into metrics per (method, gap length).

    MAE confidence intervals are 95% percentile bootstrap intervals over
    gaps (resampling whole gaps, since points within a gap are correlated).

    Returns:
        DataFrame with ``MAE``, ``MAE_lo``, ``MAE_hi``, ``RMSE``, ``MaxAE``,
        ``bias``, ``gaps``, ``points`` and mean fill time ``seconds``.
    """
    rng = np.random.default_rng(seed)
    out = []
    for (method, L), g in per_gap.groupby(["method", "gap_len"], sort=False):
        n = g["n"].to_numpy()
        gap_mae = g["sum_abs"].to_numpy() / n
        boot = rng.choice(gap_mae, size=(n_boot, len(gap_mae))).mean(axis=1)
        out.append({
            "method": method, "gap_len": L, "gap_hours": L / 6,
            "MAE": g["sum_abs"].sum() / n.sum(),
            "MAE_lo": np.percentile(boot, 2.5), "MAE_hi": np.percentile(boot, 97.5),
            "RMSE": np.sqrt(g["sum_sq"].sum() / n.sum()),
            "MaxAE": g["max_abs"].max(),
            "bias": g["sum_err"].sum() / n.sum(),
            "gaps": len(g), "points": int(n.sum()),
            "seconds": g.groupby("repeat")["seconds"].first().mean(),
        })
    return pd.DataFrame(out)
