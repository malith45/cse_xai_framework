import numpy as np
import pytest
from arch import arch_model

from cse_risk_engine import (
    GJRGARCHParams, GJRGARCHFit, ConvergenceError, fit_gjr_garch,
    filter_variance, next_variance, log_returns_pct,
)
from cse_risk_engine.simulation import simulate_gjr_garch_series


# --------------------------------------------------------------------------- #
# Parameter recovery from a process with KNOWN parameters
# --------------------------------------------------------------------------- #
def test_parameter_recovery_within_tolerance(fitted_full, true_params):
    est = fitted_full.params
    se = fitted_full.std_errors
    # Each estimate must lie within 3 standard errors of the truth AND within
    # an absolute tolerance, so the test is not trivially satisfied by huge s.e.
    abs_tol = {"omega": 0.015, "alpha": 0.03, "gamma": 0.06, "beta": 0.05, "nu": 2.0}
    for name in ("omega", "alpha", "gamma", "beta", "nu"):
        true_v, est_v = getattr(true_params, name), getattr(est, name)
        assert abs(est_v - true_v) < 3.0 * se[name], f"{name}: est={est_v:.4f} true={true_v:.4f} se={se[name]:.4f}"
        assert abs(est_v - true_v) < abs_tol[name], f"{name}: est={est_v:.4f} true={true_v:.4f}"
    assert fitted_full.converged is True
    assert fitted_full.n_obs == 5000
    assert np.isfinite(fitted_full.log_likelihood)
    assert all(np.isfinite(v) for v in se.values())


def test_recovery_is_robust_to_seed():
    """Recovery should hold for several independent synthetic series, not one lucky draw."""
    true = GJRGARCHParams(omega=0.05, alpha=0.03, gamma=0.12, beta=0.80, nu=5.0)
    for seed in (1, 2, 3):
        r = simulate_gjr_garch_series(true, n_obs=8000, seed=seed)
        f = fit_gjr_garch(r, window=None)
        se = f.std_errors
        for name, tol in (("beta", 0.08), ("gamma", 0.06), ("alpha", 0.05), ("nu", 1.5), ("omega", 0.03)):
            err = abs(getattr(f.params, name) - getattr(true, name))
            assert err < tol, f"seed {seed} {name}: err={err:.4f}"
            assert err < 3 * se[name], f"seed {seed} {name}: err={err:.4f} se={se[name]:.4f}"
        assert f.params.gamma > 2 * se["gamma"]  # leverage effect must be detected (significant)
        # persistence is pinned down much more tightly than the individual terms
        assert abs(f.params.persistence - true.persistence) < 0.03


# --------------------------------------------------------------------------- #
# Recursion consistency with the arch library
# --------------------------------------------------------------------------- #
def test_next_variance_matches_arch_one_step_forecast(synthetic_returns):
    r = synthetic_returns[-750:]
    res = arch_model(r, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False).fit(disp="off")
    p = GJRGARCHParams(res.params["omega"], res.params["alpha[1]"], res.params["gamma[1]"],
                       res.params["beta[1]"], res.params["nu"])
    ours = next_variance(p, r[-1], res.conditional_volatility[-1] ** 2)
    theirs = float(res.forecast(horizon=1).variance.iloc[-1, 0])
    assert ours == pytest.approx(theirs, rel=1e-10)


def test_fit_state_matches_arch(synthetic_returns, fitted):
    r = synthetic_returns[-750:]
    res = arch_model(r, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False).fit(disp="off")
    assert fitted.last_return == r[-1]
    assert fitted.last_sigma2 == pytest.approx(res.conditional_volatility[-1] ** 2, rel=1e-6)
    assert fitted.next_sigma2 == pytest.approx(float(res.forecast(horizon=1).variance.iloc[-1, 0]), rel=1e-6)


def test_filter_variance_reproduces_arch_conditional_variance(synthetic_returns):
    r = synthetic_returns[-750:]
    res = arch_model(r, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False).fit(disp="off")
    p = GJRGARCHParams(res.params["omega"], res.params["alpha[1]"], res.params["gamma[1]"],
                       res.params["beta[1]"], res.params["nu"])
    cv = res.conditional_volatility ** 2
    ours = filter_variance(p, r, sigma2_0=float(cv[0]))
    np.testing.assert_allclose(ours, cv, rtol=1e-8)


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def test_default_window_uses_last_750(synthetic_returns, fitted):
    assert fitted.n_obs == 750
    assert fitted.window == (len(synthetic_returns) - 750, len(synthetic_returns))
    assert fitted.last_return == synthetic_returns[-1]


def test_end_parameter_walks_window(synthetic_returns):
    f = fit_gjr_garch(synthetic_returns, window=750, end=2000)
    assert f.window == (1250, 2000)
    assert f.last_return == synthetic_returns[1999]


def test_window_too_large_raises(synthetic_returns):
    with pytest.raises(ValueError):
        fit_gjr_garch(synthetic_returns[:500], window=750)


# --------------------------------------------------------------------------- #
# Input validation and explicit non-convergence handling
# --------------------------------------------------------------------------- #
def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        fit_gjr_garch(np.zeros(800), window=None)             # zero variance
    bad = np.random.default_rng(0).normal(size=800)
    bad[10] = np.nan
    with pytest.raises(ValueError):
        fit_gjr_garch(bad, window=None)                        # NaN
    with pytest.raises(ValueError):
        fit_gjr_garch(np.random.default_rng(0).normal(size=50), window=None)  # too few


def test_decimal_input_triggers_unit_warning(synthetic_returns):
    with pytest.warns(UserWarning, match="PERCENT"):
        fit_gjr_garch(synthetic_returns[-750:] / 100.0, window=None)


class _FakeRes:
    def __init__(self, flag, params=None):
        import pandas as pd
        self.convergence_flag = flag
        vals = params or dict(omega=0.02, alpha=0.05, gamma=0.10, beta=0.85, nu=6.0)
        idx = ["omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"]
        self.params = pd.Series(list(vals.values()), index=idx)
        self.std_err = pd.Series([0.01] * 5, index=idx)
        self.loglikelihood, self.aic, self.bic = -1.0, 2.0, 3.0
        self.conditional_volatility = np.ones(750)


class _FakeModel:
    def __init__(self, res):
        self._res = res

    def fit(self, *a, **k):
        return self._res


def test_nonconvergence_raises_explicitly(monkeypatch, synthetic_returns):
    import arch
    monkeypatch.setattr(arch, "arch_model", lambda *a, **k: _FakeModel(_FakeRes(flag=1)))
    with pytest.raises(ConvergenceError, match="convergence_flag=1"):
        fit_gjr_garch(synthetic_returns, window=750, max_attempts=2)


def test_optimiser_exception_becomes_convergence_error(monkeypatch, synthetic_returns):
    import arch

    class _Boom:
        def fit(self, *a, **k):
            raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(arch, "arch_model", lambda *a, **k: _Boom())
    with pytest.raises(ConvergenceError, match="LinAlgError"):
        fit_gjr_garch(synthetic_returns, window=750)


def test_nonstationary_fit_rejected_when_strict(monkeypatch, synthetic_returns):
    import arch
    bad = dict(omega=0.02, alpha=0.20, gamma=0.10, beta=0.85, nu=6.0)  # persistence 1.10
    monkeypatch.setattr(arch, "arch_model", lambda *a, **k: _FakeModel(_FakeRes(0, bad)))
    with pytest.raises(ConvergenceError, match="non-stationary"):
        fit_gjr_garch(synthetic_returns, window=750)
    f = fit_gjr_garch(synthetic_returns, window=750, strict=False)
    assert not f.params.is_stationary and any("non-stationary" in w for w in f.warnings)


def test_invalid_parameter_values_rejected():
    with pytest.raises(ValueError):
        GJRGARCHParams(omega=0.0, alpha=0.05, gamma=0.1, beta=0.85, nu=6)
    with pytest.raises(ValueError):
        GJRGARCHParams(omega=0.02, alpha=0.05, gamma=0.1, beta=0.85, nu=2.0)
    with pytest.raises(ValueError):
        GJRGARCHParams(omega=0.02, alpha=0.05, gamma=-0.1, beta=0.85, nu=6)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_log_returns_pct():
    prices = np.array([100.0, 101.0, 99.0])
    r = log_returns_pct(prices)
    np.testing.assert_allclose(r, 100 * np.log(prices[1:] / prices[:-1]))
    assert r[0] == pytest.approx(0.99503, abs=1e-5)
    with pytest.raises(ValueError):
        log_returns_pct([100.0, 0.0])


def test_derived_quantities(true_params):
    assert true_params.persistence == pytest.approx(0.05 + 0.05 + 0.85)
    assert true_params.unconditional_variance == pytest.approx(0.02 / (1 - 0.95))
    assert true_params.is_stationary
