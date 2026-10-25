"""Verification of the gap-filling methods against known answers.

Includes convergence-order tests: on a smooth function, halving the
sample spacing h should cut the interpolation error by 2^order, where the
order is 2 for linear interpolation and 4 for a cubic spline. Matching
the theoretical order is strong evidence the method is implemented
correctly, not just that it runs.
"""
import numpy as np
import pytest

from conftest import make_frame
from tempfill.methods import (BASIC_METHODS, BoostedFill, fill_cubic_spline, fill_harmonic,
                              fill_hybrid, fill_linear, fill_linear_loop, fill_pchip,
                              harmonic_design, solve_least_squares)

ALL = {**BASIC_METHODS, "hybrid": fill_hybrid}


def _midpoint_error(method, omega):
    """Max error filling every other slot of sin(omega * slot)."""
    x = np.arange(400)
    y = np.sin(omega * x)
    masked = y.copy()
    masked[1:-1:2] = np.nan            # knots spaced 2 slots apart
    filled = method(make_frame(masked))
    inner = slice(50, 350)             # avoid boundary effects
    return np.abs(filled - y)[inner].max()


@pytest.mark.parametrize("method, order", [(fill_linear, 2), (fill_cubic_spline, 4)])
def test_convergence_order(method, order):
    errors = [_midpoint_error(method, w) for w in (0.2, 0.1, 0.05)]
    observed = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert observed == pytest.approx([order, order], abs=0.15)


@pytest.mark.parametrize("name", list(ALL))
def test_observed_values_unchanged_and_no_nan(name, daily_frame):
    f = daily_frame.copy()
    f.loc[[0, 1, 300, 301, 302, 2000, len(f) - 1], "temperature"] = np.nan  # incl. edges
    filled = ALL[name](f)
    assert np.isfinite(filled).all()
    obs = f["temperature"].notna().to_numpy()
    np.testing.assert_array_equal(filled[obs], f["temperature"].to_numpy()[obs])


def test_linear_exact_on_straight_line():
    y = 20 + 0.01 * np.arange(50)
    m = y.copy()
    m[10:30] = np.nan
    np.testing.assert_allclose(fill_linear(make_frame(m)), y, atol=1e-12)


def test_spline_exact_on_cubic():
    x = np.arange(60.0)
    y = 1e-5 * (x - 30) ** 3 + 0.01 * x + 20
    m = y.copy()
    m[20:25] = np.nan
    np.testing.assert_allclose(fill_cubic_spline(make_frame(m)), y, atol=1e-9)


def test_pchip_does_not_overshoot_a_plateau():
    """Heating ramps up then holds (HVAC reaches setpoint); gap over the corner."""
    y = np.r_[np.linspace(15, 25, 21), np.full(20, 25.0)]
    m = y.copy()
    m[18:30] = np.nan
    assert fill_pchip(make_frame(m)).max() <= 25.0 + 1e-12
    assert fill_cubic_spline(make_frame(m)).max() > 25.05   # the spline does overshoot


def test_loop_matches_vectorised():
    rng = np.random.default_rng(1)
    y = rng.normal(20, 1, 500)
    y[rng.random(500) < 0.3] = np.nan
    y[[0, -1]] = np.nan
    f = make_frame(y)
    np.testing.assert_allclose(fill_linear_loop(f), fill_linear(f))


def test_harmonic_recovers_exact_sinusoid(daily_frame):
    """A 1-day gap in a pure daily/weekly signal is reconstructed exactly."""
    m = daily_frame.copy()
    m.loc[700:843, "temperature"] = np.nan
    filled = fill_harmonic(m, k_daily=2, k_weekly=1, window_days=5)
    np.testing.assert_allclose(filled, daily_frame["temperature"], atol=1e-8)
    assert np.abs(fill_linear(m) - daily_frame["temperature"]).max() > 1.0  # linear cannot


def test_harmonic_level_shift_is_absorbed(daily_frame):
    """A constant offset is handled by the residual bridge, not the fit."""
    shifted = daily_frame.copy()
    shifted["temperature"] += 3.0
    m = shifted.copy()
    m.loc[700:760, "temperature"] = np.nan
    np.testing.assert_allclose(fill_harmonic(m, k_daily=2, k_weekly=1), shifted["temperature"],
                               atol=1e-8)


def test_hybrid_uses_linear_for_short_gaps(daily_frame):
    m = daily_frame.copy()
    m.loc[[100, 101], "temperature"] = np.nan
    np.testing.assert_allclose(fill_hybrid(m, switch_len=6), fill_linear(m))


def test_design_matrix_shape(daily_frame):
    X = harmonic_design(daily_frame["time"], k_daily=3, k_weekly=2)
    assert X.shape == (len(daily_frame), 2 + 6 + 4)
    np.testing.assert_allclose(X[:, 0], 1.0)


@pytest.mark.parametrize("solver", ["lstsq", "normal"])
def test_solvers_agree_on_well_conditioned_problem(solver):
    rng = np.random.default_rng(0)
    A = rng.normal(size=(200, 5))
    c = np.arange(1.0, 6.0)
    np.testing.assert_allclose(solve_least_squares(A, A @ c, solver), c, atol=1e-10)


def test_unknown_solver_raises():
    with pytest.raises(ValueError):
        solve_least_squares(np.eye(3), np.ones(3), "magic")


def test_too_few_observations_raises():
    with pytest.raises(ValueError):
        fill_linear(make_frame([np.nan, 1.0, np.nan]))


def test_boosted_fill_runs_and_is_reasonable(daily_frame):
    rng = np.random.default_rng(0)
    f = daily_frame.copy()
    f["temperature"] += rng.normal(0, 0.05, len(f))
    model = BoostedFill(max_iter=30).fit(f, (0, len(f)), lengths=(3, 36), n_gaps=10, repeats=1)
    m = f.copy()
    m.loc[1000:1035, "temperature"] = np.nan
    filled = model(m)
    assert np.isfinite(filled).all()
    err = np.abs(filled - f["temperature"].to_numpy())[1000:1036].mean()
    assert err < 0.5


def test_boosted_rejects_bad_base():
    with pytest.raises(ValueError):
        BoostedFill(base="spline")
