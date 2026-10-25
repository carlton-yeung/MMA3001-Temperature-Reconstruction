"""Tests for synthetic masking, metrics and performance helpers."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_frame
from tempfill.methods import fill_linear, fill_linear_loop
from tempfill.performance import (flop_estimates, gap_count_scaling, optimisation_steps,
                                  synthetic_frame, vectorisation_speedup)
from tempfill.validation import (adopt_hybrid, apply_gaps, choose_switch, error_metrics,
                                 run_gap_experiment, sample_gaps, simplest_within, summarise,
                                 usable_horizon)


def test_sample_gaps_rules():
    rng = np.random.default_rng(0)
    observed = np.ones(2000, dtype=bool)
    observed[::13] = False                       # realistic single dropouts
    L = 20
    starts = sample_gaps(observed, L, 30, rng, region=(500, 1800))
    assert len(starts) > 0
    assert (starts >= 501).all() and (starts + L <= 1799).all()   # inside region
    assert observed[starts - 1].all() and observed[starts + L].all()  # anchors observed
    assert (np.diff(starts) >= 2 * L).all()                       # separated, no overlap
    for s in starts:
        assert observed[s:s + L].mean() >= 0.8


def test_sample_gaps_reproducible():
    obs = np.ones(1000, dtype=bool)
    a = sample_gaps(obs, 10, 5, np.random.default_rng(3))
    b = sample_gaps(obs, 10, 5, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)


def test_sample_gaps_invalid_length():
    with pytest.raises(ValueError):
        sample_gaps(np.ones(10, dtype=bool), 0, 1, np.random.default_rng())


def test_apply_gaps_mask_only_where_truth_known():
    y = np.arange(20.0)
    y[7] = np.nan
    masked, mask = apply_gaps(make_frame(y), [5], 4)
    assert np.isnan(masked["temperature"].iloc[5:9]).all()
    assert mask.tolist()[5:9] == [True, True, False, True]
    assert not np.isnan(make_frame(y)["temperature"].iloc[5])   # original untouched


def test_error_metrics_known_values():
    m = error_metrics(np.array([1.0, -1.0, 2.0, -2.0]))
    assert m["MAE"] == 1.5 and m["bias"] == 0.0
    assert m["RMSE"] == pytest.approx(np.sqrt(2.5)) and m["MaxAE"] == 2.0


def test_experiment_zero_error_on_linear_signal():
    f = make_frame(20 + 0.01 * np.arange(3000))
    per_gap = run_gap_experiment(f, {"linear": fill_linear}, [6, 36], n_gaps=5, repeats=2)
    s = summarise(per_gap)
    assert set(s["gap_len"]) == {6, 36}
    assert s["MAE"].max() < 1e-10
    assert (s["MAE_lo"] <= s["MAE"]).all() and (s["MAE"] <= s["MAE_hi"] + 1e-12).all()


def test_synthetic_frame_properties():
    f = synthetic_frame(5000, missing_fraction=0.1, seed=1)
    miss = f["temperature"].isna().mean()
    assert 0.08 < miss < 0.2
    assert np.isfinite(f["temperature"].iloc[[0, -1]]).all()


def test_vectorisation_speedup_reports_identical_output():
    out = vectorisation_speedup(synthetic_frame(20_000), fill_linear_loop, fill_linear, repeats=1)
    assert out["identical_output"] and out["speedup"] > 1


def test_flop_estimates_hybrid_cheaper_than_harmonic():
    t = flop_estimates(synthetic_frame(10_000)).set_index("method")
    assert t.loc["hybrid", "flops"] < t.loc["harmonic", "flops"]


def test_fully_observed_design_has_no_missing_inside():
    rng = np.random.default_rng(0)
    observed = np.ones(3000, dtype=bool)
    observed[::9] = False
    starts = sample_gaps(observed, 6, 30, rng, min_observed=1.0)
    assert len(starts) > 0
    for s in starts:
        assert observed[s - 1:s + 7].all()


def test_keep_points_gives_p95():
    f = synthetic_frame(4000, seed=3)
    pg, pts = run_gap_experiment(f, {"linear": fill_linear}, [6], n_gaps=5, repeats=2,
                                 keep_points=True)
    s = summarise(pg, pts)
    assert "P95AE" in s and s["P95AE"].iloc[0] >= s["MAE"].iloc[0]
    assert len(pts[("linear", 6)]) == pg["n"].sum()


def _summary(rows):
    """Build a minimal summary table from (method, gap_len, MAE) tuples."""
    df = pd.DataFrame(rows, columns=["method", "gap_len", "MAE"])
    df["gap_hours"] = df["gap_len"] / 6
    return df


def test_choose_switch_stops_at_first_loss():
    s = _summary([("linear", 1, 0.10), ("harmonic", 1, 0.10),
                  ("linear", 6, 0.20), ("harmonic", 6, 0.199),   # within 2%
                  ("linear", 18, 0.40), ("harmonic", 18, 0.30),  # loses
                  ("linear", 36, 0.50), ("harmonic", 36, 0.51)])  # irrelevant: after a loss
    assert choose_switch(s) == 6


def test_adopt_hybrid_needs_gain_and_no_loss():
    base = [("linear", 1, 0.10), ("linear", 72, 1.00),
            ("harmonic", 1, 0.20), ("harmonic", 72, 0.50)]
    gain = _summary(base + [("hybrid", 1, 0.10), ("hybrid", 72, 0.50)])
    assert adopt_hybrid(gain) == (True, "harmonic")       # 0.10 better at 10 min
    small = _summary(base[:2] + [("harmonic", 1, 0.15), ("harmonic", 72, 0.50),
                                 ("hybrid", 1, 0.10), ("hybrid", 72, 0.50)])
    assert adopt_hybrid(small)[0] is False                # only 0.05 better


def test_simplest_within_prefers_simple():
    s = _summary([("linear", 1, 0.100), ("harmonic", 1, 0.099)])
    assert simplest_within(s, ["linear", "harmonic"]) == "linear"


def test_usable_horizon_is_conservative_and_contiguous():
    s = pd.DataFrame({"method": "m", "gap_len": [1, 6, 36, 72],
                      "gap_hours": [1 / 6, 1, 6, 12],
                      "MAE_hi": [0.1, 0.45, 0.55, 0.40]})   # 12 h passes but 6 h fails
    assert usable_horizon(s, 0.5)["m"] == 1


def test_gap_count_scaling_and_optimisation_steps():
    from tempfill.methods import fill_harmonic, fill_harmonic_naive, fill_hybrid
    t, slope = gap_count_scaling(fill_hybrid, n=8000, fractions=(0.02, 0.1), repeats=1)
    assert len(t) == 2 and np.isfinite(slope)
    f = synthetic_frame(4000, seed=5)
    tab = optimisation_steps(f, {"naive": fill_harmonic_naive, "opt": fill_harmonic},
                             "naive", repeats=1)
    assert tab["max_diff_C"].max() < 1e-8
