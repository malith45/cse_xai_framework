import time

import numpy as np
import pytest

from cse_risk_engine import (
    GJRGARCHParams, simulate_paths, next_variance, estimate_var, run_stress_test,
    VolatilityMultiplier, ReturnShock,
)
from cse_risk_engine.simulation import standardised_t, simulate_gjr_garch_series


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def test_same_seed_identical_output(fitted):
    a = simulate_paths(fitted, horizon=5, n_paths=2000, seed=7)
    b = simulate_paths(fitted, horizon=5, n_paths=2000, seed=7)
    np.testing.assert_array_equal(a.returns, b.returns)
    np.testing.assert_array_equal(a.sigma2, b.sigma2)
    v1 = estimate_var(fitted, 1e6, seed=7)
    v2 = estimate_var(fitted, 1e6, seed=7)
    assert v1.table() == v2.table()


def test_different_seed_different_output(fitted):
    a = simulate_paths(fitted, horizon=1, n_paths=2000, seed=1)
    b = simulate_paths(fitted, horizon=1, n_paths=2000, seed=2)
    assert not np.array_equal(a.returns, b.returns)


# --------------------------------------------------------------------------- #
# Performance target: 10,000-path stress test < 10 s
# --------------------------------------------------------------------------- #
def test_ten_thousand_path_stress_test_under_ten_seconds(fitted):
    t0 = time.perf_counter()
    baseline = estimate_var(fitted, 1e6, horizons=(1, 5), n_paths=10_000, seed=0)
    run_stress_test(fitted, VolatilityMultiplier(2.0), 1e6, n_paths=10_000, seed=0, baseline=baseline)
    run_stress_test(fitted, ReturnShock(-5.0), 1e6, n_paths=10_000, seed=0, baseline=baseline)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"took {elapsed:.2f}s"
    # and comfortably so: a single 10k x 5 simulation should be well under a second
    t0 = time.perf_counter()
    simulate_paths(fitted, horizon=5, n_paths=10_000, seed=0)
    assert time.perf_counter() - t0 < 1.0


# --------------------------------------------------------------------------- #
# Asymmetry: the gamma (leverage) term
# --------------------------------------------------------------------------- #
def test_negative_shock_raises_variance_more_than_positive(true_params):
    s2 = true_params.unconditional_variance
    neg = next_variance(true_params, -2.0, s2)
    pos = next_variance(true_params, +2.0, s2)
    assert neg > pos
    # exact size of the asymmetry: gamma * r^2
    assert neg - pos == pytest.approx(true_params.gamma * 4.0)


def test_asymmetry_propagates_through_simulation(fitted):
    """Mean simulated variance over days 1..5 must be higher after a negative
    initial return than after a positive one of equal magnitude."""
    shock = 3.0
    neg = simulate_paths(fitted, horizon=5, n_paths=20_000, seed=11, initial_return=-shock)
    pos = simulate_paths(fitted, horizon=5, n_paths=20_000, seed=11, initial_return=+shock)
    assert fitted.params.gamma > 0
    assert neg.sigma2[:, 0].mean() > pos.sigma2[:, 0].mean()
    assert neg.sigma2.mean() > pos.sigma2.mean()
    # and the 5-day loss tail is fatter
    assert np.percentile(neg.cumulative, 5) < np.percentile(pos.cumulative, 5)


def test_no_asymmetry_when_gamma_is_zero():
    p = GJRGARCHParams(omega=0.02, alpha=0.10, gamma=0.0, beta=0.85, nu=6)
    s2 = p.unconditional_variance
    assert next_variance(p, -2.0, s2) == next_variance(p, +2.0, s2)


def test_vectorised_next_variance_matches_scalar(true_params):
    r = np.array([-1.0, 0.5, -2.5, 3.0])
    s2 = np.array([0.4, 0.5, 0.6, 0.7])
    vec = next_variance(true_params, r, s2)
    for i in range(4):
        assert vec[i] == pytest.approx(next_variance(true_params, r[i], s2[i]))


# --------------------------------------------------------------------------- #
# Distributional and structural checks
# --------------------------------------------------------------------------- #
def test_standardised_t_has_unit_variance():
    rng = np.random.default_rng(0)
    z = standardised_t(rng, 6.0, size=2_000_000)
    assert z.var() == pytest.approx(1.0, abs=0.01)
    assert z.mean() == pytest.approx(0.0, abs=0.01)


def test_shapes_and_cumulative(fitted):
    sim = simulate_paths(fitted, horizon=5, n_paths=1234, seed=3)
    assert sim.returns.shape == (1234, 5)
    assert sim.sigma2.shape == (1234, 5)
    assert sim.cumulative.shape == (1234,)
    np.testing.assert_allclose(sim.cumulative, sim.returns.sum(axis=1))
    assert np.all(sim.sigma2 > 0)


def test_first_day_variance_is_deterministic_and_equals_next_sigma2(fitted):
    sim = simulate_paths(fitted, horizon=3, n_paths=500, seed=5)
    assert np.allclose(sim.sigma2[:, 0], fitted.next_sigma2)
    assert sim.initial_sigma2 == pytest.approx(fitted.next_sigma2)
    # simulated day-2 variance obeys eq. 4.5 path by path
    expected = next_variance(fitted.params, sim.returns[:, 0], sim.sigma2[:, 0])
    np.testing.assert_allclose(sim.sigma2[:, 1], expected)


def test_sigma2_multiplier_scales_first_day_variance(fitted):
    base = simulate_paths(fitted, horizon=1, n_paths=100, seed=1)
    stressed = simulate_paths(fitted, horizon=1, n_paths=100, seed=1, sigma2_multiplier=4.0)
    np.testing.assert_allclose(stressed.sigma2[:, 0], 4.0 * base.sigma2[:, 0])
    np.testing.assert_allclose(stressed.returns, 2.0 * base.returns)


def test_bare_params_require_initial_state(true_params):
    with pytest.raises(ValueError):
        simulate_paths(true_params, horizon=1, n_paths=10)
    sim = simulate_paths(true_params, horizon=1, n_paths=10, seed=0, initial_return=0.0, initial_sigma2=0.4)
    assert sim.returns.shape == (10, 1)


def test_synthetic_generator_matches_unconditional_variance(true_params):
    r = simulate_gjr_garch_series(true_params, n_obs=200_000, seed=99)
    assert r.var() == pytest.approx(true_params.unconditional_variance, rel=0.05)
