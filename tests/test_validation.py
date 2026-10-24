"""Tests for synthetic masking, metrics and performance helpers."""
import numpy as np
import pytest

from conftest import make_frame
from tempfill.methods import fill_linear, fill_linear_loop
from tempfill.performance import flop_estimates, synthetic_frame, vectorisation_speedup
from tempfill.validation import (apply_gaps, error_metrics, run_gap_experiment, sample_gaps,
                                 summarise)


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
