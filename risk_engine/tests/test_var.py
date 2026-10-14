import numpy as np
import pytest
from scipy.stats import norm, t as student_t

from cse_risk_engine import GJRGARCHParams, estimate_var, value_at_risk, expected_shortfall
from cse_risk_engine.var import log_loss_to_lkr

# A "degenerate" GJR-GARCH with alpha = beta = gamma = 0 and omega = sigma²
# produces i.i.d. returns with variance sigma² — so the analytic VaR is known.
SIGMA = 1.2  # percent


def _iid_params(nu):
    return GJRGARCHParams(omega=SIGMA ** 2, alpha=0.0, gamma=0.0, beta=0.0, nu=nu)


# --------------------------------------------------------------------------- #
# Monte Carlo VaR vs analytic VaR for known distributions
# --------------------------------------------------------------------------- #
def test_mc_var_matches_analytic_normal_var():
    # nu = 1e7 makes the standardised Student-t indistinguishable from N(0,1)
    p = _iid_params(nu=1e7)
    res = estimate_var(p, notional=1.0, horizons=(1, 5), confidence=(0.95, 0.99),
                       n_paths=400_000, seed=123, initial_return=0.0, initial_sigma2=SIGMA ** 2)
    for h in (1, 5):
        sig_h = SIGMA * np.sqrt(h)          # iid: h-day variance = h * sigma²
        for c in (0.95, 0.99):
            z = norm.ppf(c)
            analytic_var = z * sig_h
            analytic_es = sig_h * norm.pdf(z) / (1 - c)
            e = res.get(h, c)
            assert e.var_pct == pytest.approx(analytic_var, rel=0.02), (h, c)
            assert e.es_pct == pytest.approx(analytic_es, rel=0.03), (h, c)


def test_mc_var_matches_analytic_student_t_var():
    """Checks the (nu-2)/nu rescaling: VaR = t_ppf(c, nu) * sqrt((nu-2)/nu) * sigma."""
    nu = 5.0
    p = _iid_params(nu=nu)
    res = estimate_var(p, notional=1.0, horizons=(1,), confidence=(0.95, 0.99),
                       n_paths=400_000, seed=321, initial_return=0.0, initial_sigma2=SIGMA ** 2)
    for c in (0.95, 0.99):
        analytic = student_t.ppf(c, nu) * np.sqrt((nu - 2) / nu) * SIGMA
        assert res.get(1, c).var_pct == pytest.approx(analytic, rel=0.02)
    # fat tails: 99 % t-VaR exceeds normal 99 % VaR, 95 % t-VaR is below normal 95 % VaR
    assert res.get(1, 0.99).var_pct > norm.ppf(0.99) * SIGMA
    assert res.get(1, 0.95).var_pct < norm.ppf(0.95) * SIGMA


# --------------------------------------------------------------------------- #
# Hand-checked quantile / ES arithmetic
# --------------------------------------------------------------------------- #
def test_value_at_risk_and_es_by_hand():
    x = np.linspace(-99.0, 0.0, 100)           # -99, -98, ..., 0
    # numpy linear interpolation: 5th percentile sits at index 0.05*99 = 4.95 -> -94.05
    assert value_at_risk(x, 0.95) == pytest.approx(94.05)
    # ES: mean of returns <= -94.05 -> {-99,...,-95}, mean -97
    assert expected_shortfall(x, 0.95) == pytest.approx(97.0)
    assert value_at_risk(x, 0.99) == pytest.approx(98.01)


def test_es_never_below_var(fitted):
    res = estimate_var(fitted, 1e6, seed=9)
    for e in res.estimates.values():
        assert e.es_pct >= e.var_pct
        assert e.es_lkr >= e.var_lkr
    # 99 % VaR >= 95 % VaR, 5-day VaR > 1-day VaR
    assert res.get(1, 0.99).var_pct > res.get(1, 0.95).var_pct
    assert res.get(5, 0.95).var_pct > res.get(1, 0.95).var_pct


# --------------------------------------------------------------------------- #
# Units: percent log-return VaR -> LKR
# --------------------------------------------------------------------------- #
def test_lkr_conversion_is_exact_log_return_loss():
    notional = 1_000_000.0
    assert log_loss_to_lkr(10.0, notional) == pytest.approx(notional * (1 - np.exp(-0.10)))
    assert log_loss_to_lkr(10.0, notional) == pytest.approx(95_162.58, abs=0.01)
    # exact loss is always slightly BELOW the linear approximation for positive VaR
    assert log_loss_to_lkr(10.0, notional) < 100_000.0
    assert log_loss_to_lkr(1.0, notional) == pytest.approx(9_950.17, abs=0.01)
    assert log_loss_to_lkr(0.0, notional) == 0.0


def test_var_result_carries_notional_and_scales(fitted):
    a = estimate_var(fitted, 1e6, seed=5)
    b = estimate_var(fitted, 2e6, seed=5)
    for k in a.estimates:
        assert a.get(*k).var_pct == b.get(*k).var_pct        # percent is notional-free
        assert b.get(*k).var_lkr == pytest.approx(2 * a.get(*k).var_lkr)
    assert set(a.estimates) == {(1, 0.95), (1, 0.99), (5, 0.95), (5, 0.99)}
    d = a.get(1, 0.95).as_dict()
    assert d["horizon_days"] == 1 and d["notional_lkr"] == 1e6


def test_input_validation():
    with pytest.raises(ValueError):
        value_at_risk(np.array([1.0, 2.0]), 1.5)
    with pytest.raises(ValueError):
        value_at_risk(np.array([]), 0.95)
