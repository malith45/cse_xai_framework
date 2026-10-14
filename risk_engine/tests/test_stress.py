import numpy as np
import pytest

from cse_risk_engine import (
    GJRGARCHParams, run_stress_test, estimate_var,
    VolatilityMultiplier, ReturnShock, CrisisScenario, CrisisCalibration, calibrate_crisis,
)
from cse_risk_engine.stress import PLACEHOLDER_LABEL
from cse_risk_engine.simulation import simulate_gjr_garch_series


def test_volatility_multiplier_scales_one_day_var_exactly(fitted):
    """With the same seed the 1-day returns are sigma * z, so 1-day VaR scales
    by exactly the multiplier.  5-day VaR scales by roughly (not exactly) the
    multiplier because the recursion pulls variance back toward omega."""
    res = run_stress_test(fitted, VolatilityMultiplier(2.0), 1e6, seed=42)
    for c in (0.95, 0.99):
        assert res.uplift(1, c) == pytest.approx(2.0, rel=1e-9)
        assert 1.5 < res.uplift(5, c) < 2.0
    assert res.is_placeholder is False
    assert res.stressed.get(1, 0.95).var_lkr > res.baseline.get(1, 0.95).var_lkr


def test_volatility_multiplier_below_one_reduces_var(fitted):
    res = run_stress_test(fitted, VolatilityMultiplier(0.5), 1e6, seed=42)
    assert res.uplift(1, 0.95) == pytest.approx(0.5, rel=1e-9)


def test_return_shock_raises_var_and_is_monotone(fitted):
    base = estimate_var(fitted, 1e6, seed=42)
    small = run_stress_test(fitted, ReturnShock(-3.0), 1e6, seed=42, baseline=base)
    large = run_stress_test(fitted, ReturnShock(-8.0), 1e6, seed=42, baseline=base)
    for h in (1, 5):
        for c in (0.95, 0.99):
            assert small.stressed.get(h, c).var_pct > base.get(h, c).var_pct
            assert large.stressed.get(h, c).var_pct > small.stressed.get(h, c).var_pct
    # the shock enters through the (alpha + gamma) term: day-1 variance is exactly eq. 4.5
    p = fitted.params
    expected_s2 = p.omega + (p.alpha + p.gamma) * 9.0 + p.beta * fitted.last_sigma2
    ratio = small.stressed.get(1, 0.95).var_pct / base.get(1, 0.95).var_pct
    assert ratio == pytest.approx(np.sqrt(expected_s2 / fitted.next_sigma2), rel=1e-9)


def test_return_shock_asymmetry(fitted):
    neg = run_stress_test(fitted, ReturnShock(-4.0), 1e6, seed=42)
    with pytest.warns(UserWarning):
        pos_scn = ReturnShock(+4.0)
    pos = run_stress_test(fitted, pos_scn, 1e6, seed=42)
    assert neg.stressed.get(1, 0.95).var_pct > pos.stressed.get(1, 0.95).var_pct


def test_placeholder_crisis_is_clearly_flagged(fitted):
    cal = CrisisCalibration.placeholder(fit=fitted)
    assert cal.is_placeholder
    assert cal.params == fitted.params
    assert "PLACEHOLDER" in cal.label
    res = run_stress_test(fitted, CrisisScenario(cal), 1e6, seed=42)
    assert res.is_placeholder
    assert any(PLACEHOLDER_LABEL in n for n in res.notes)
    assert "PLACEHOLDER" in str(res)


def test_crisis_calibration_interface_with_synthetic_crisis(fitted):
    """A crisis period simulated with much higher volatility must produce a
    non-placeholder calibration and a materially larger VaR."""
    crisis_true = GJRGARCHParams(omega=0.30, alpha=0.08, gamma=0.15, beta=0.78, nu=4.5)
    crisis_returns = simulate_gjr_garch_series(crisis_true, n_obs=400, seed=2022)
    cal = calibrate_crisis(crisis_returns, label="synthetic 2022 crisis", period=("2022-01-03", "2022-12-30"),
                           source="synthetic", window=None)
    assert cal.is_placeholder is False
    assert cal.n_obs == 400 and cal.log_likelihood is not None
    assert cal.params.unconditional_variance > fitted.params.unconditional_variance
    res = run_stress_test(fitted, CrisisScenario(cal), 1e6, seed=42)
    assert res.is_placeholder is False
    assert res.uplift(1, 0.95) > 1.5
    assert res.uplift(5, 0.99) > 1.5
    # starting from the current state instead of the crisis long-run variance
    res2 = run_stress_test(fitted, CrisisScenario(cal, use_current_state=True), 1e6, seed=42)
    assert res2.stressed.get(1, 0.95).var_pct != res.stressed.get(1, 0.95).var_pct


def test_crisis_calibration_explicit_initial_sigma2(fitted):
    cal = CrisisCalibration(params=fitted.params, label="manual", initial_sigma2=9.0)
    assert cal.start_sigma2 == 9.0
    res = run_stress_test(fitted, CrisisScenario(cal), 1e6, seed=1)
    p = fitted.params
    expected_first = p.omega + p.beta * 9.0   # r0 = 0 -> indicator irrelevant
    ratio = res.stressed.get(1, 0.95).var_pct / res.baseline.get(1, 0.95).var_pct
    assert ratio == pytest.approx(np.sqrt(expected_first / fitted.next_sigma2), rel=1e-9)


def test_baseline_reuse_and_validation(fitted):
    base = estimate_var(fitted, 1e6, seed=3)
    res = run_stress_test(fitted, VolatilityMultiplier(1.5), 1e6, seed=3, baseline=base)
    assert res.baseline is base
    with pytest.raises(ValueError):
        VolatilityMultiplier(0.0)
    with pytest.raises(TypeError):
        run_stress_test(fitted, "not a scenario", 1e6)
