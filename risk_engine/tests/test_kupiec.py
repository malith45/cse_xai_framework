import math

import numpy as np
import pytest
from scipy.stats import chi2

from cse_risk_engine import kupiec_pof_test
from cse_risk_engine.kupiec import kupiec_lr_statistic

CHI2_5PCT = 3.841458820694124  # chi²(1) critical value at 5 %


# --------------------------------------------------------------------------- #
# Worked textbook example: Jorion (2007), Value at Risk, 3rd ed., ch. 6.
# JPMorgan's 1998 annual report: 20 exceptions in 252 trading days at 95 % VaR.
# Jorion reports LR_uc = 3.91, just above the 3.84 cut-off -> reject at 5 %.
# --------------------------------------------------------------------------- #
def test_jorion_jpmorgan_example():
    k = kupiec_pof_test(n_obs=252, n_breaches=20, confidence=0.95)
    assert k.lr_statistic == pytest.approx(3.91, abs=0.005)
    assert k.p_value == pytest.approx(0.0479, abs=0.001)
    assert k.passed is False
    assert k.observed_rate == pytest.approx(20 / 252)
    assert k.expected_rate == pytest.approx(0.05)
    assert k.critical_value == pytest.approx(CHI2_5PCT)


# --------------------------------------------------------------------------- #
# Kupiec (1995), Table 3 (reproduced as Jorion Table 6-3): non-rejection
# regions at 5 % significance.  For p = 0.05, T = 255: 6 < N < 21.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("x,expected_pass", [(6, False), (7, True), (20, True), (21, False)])
def test_kupiec_1995_nonrejection_region_p05_T255(x, expected_pass):
    k = kupiec_pof_test(n_obs=255, n_breaches=x, confidence=0.95)
    assert k.passed is expected_pass


@pytest.mark.parametrize("T,lo,hi", [(510, 17, 35), (1000, 38, 64)])
def test_kupiec_1995_nonrejection_region_p05_larger_T(T, lo, hi):
    # Kupiec: T=510 -> 16 < N < 36 ; T=1000 -> 37 < N < 65
    assert kupiec_pof_test(n_obs=T, n_breaches=lo, confidence=0.95).passed
    assert kupiec_pof_test(n_obs=T, n_breaches=hi, confidence=0.95).passed
    assert not kupiec_pof_test(n_obs=T, n_breaches=lo - 1, confidence=0.95).passed
    assert not kupiec_pof_test(n_obs=T, n_breaches=hi + 1, confidence=0.95).passed


def test_kupiec_1995_p01_T255_region():
    # Kupiec: p = 0.01, T = 255 -> N < 7  (upper bound of the region)
    assert kupiec_pof_test(n_obs=255, n_breaches=6, confidence=0.99).passed
    assert not kupiec_pof_test(n_obs=255, n_breaches=7, confidence=0.99).passed


# --------------------------------------------------------------------------- #
# Hand-computed cases and the formula itself
# --------------------------------------------------------------------------- #
def test_exact_expected_breaches_give_zero_statistic():
    # x/T == p exactly -> the two likelihoods coincide -> LR = 0, p-value = 1
    k = kupiec_pof_test(n_obs=100, n_breaches=5, confidence=0.95)
    assert k.lr_statistic == pytest.approx(0.0, abs=1e-12)
    assert k.p_value == pytest.approx(1.0)
    assert k.passed


def test_formula_by_hand_T100_x10_p05():
    T, x, p = 100, 10, 0.05
    expected = (-2 * math.log((1 - p) ** (T - x) * p ** x)
                + 2 * math.log((1 - x / T) ** (T - x) * (x / T) ** x))
    assert kupiec_lr_statistic(T, x, p) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(4.1308, abs=1e-3)      # hand value
    k = kupiec_pof_test(n_obs=T, n_breaches=x, confidence=0.95)
    assert not k.passed


# --------------------------------------------------------------------------- #
# Edge cases: x = 0 and x = T (raw formula takes ln 0)
# --------------------------------------------------------------------------- #
def test_zero_breaches_uses_limit_formula():
    # LR = -2 T ln(1 - p)
    for T, p in [(250, 0.01), (250, 0.05), (10, 0.05), (1000, 0.01)]:
        k = kupiec_pof_test(n_obs=T, n_breaches=0, confidence=1 - p)
        assert math.isfinite(k.lr_statistic)
        assert k.lr_statistic == pytest.approx(-2 * T * math.log(1 - p), rel=1e-12)
    # 0 breaches in 250 days at 99 %: LR = 5.025 -> too conservative, reject
    assert not kupiec_pof_test(n_obs=250, n_breaches=0, confidence=0.99).passed
    # 0 breaches in 10 days at 95 %: LR = 1.026 -> cannot reject
    assert kupiec_pof_test(n_obs=10, n_breaches=0, confidence=0.95).passed


def test_all_breaches_uses_limit_formula():
    # LR = -2 T ln p
    for T, p in [(10, 0.05), (250, 0.01), (3, 0.05)]:
        k = kupiec_pof_test(n_obs=T, n_breaches=T, confidence=1 - p)
        assert math.isfinite(k.lr_statistic)
        assert k.lr_statistic == pytest.approx(-2 * T * math.log(p), rel=1e-12)
        assert not k.passed
    assert kupiec_pof_test(n_obs=10, n_breaches=10, confidence=0.95).lr_statistic == pytest.approx(59.915, abs=0.01)


def test_limits_are_continuous_with_general_formula():
    """The explicit limits must equal what the general formula approaches."""
    T, p = 500, 0.05
    # x = 0 vs x -> 0+: compare with a tiny x using the general formula
    lr0 = kupiec_lr_statistic(T, 0, p)
    assert lr0 == pytest.approx(-2 * T * math.log(1 - p))
    # monotone: LR should rise as x moves away from T*p in either direction
    vals = [kupiec_lr_statistic(T, x, p) for x in range(0, T + 1)]
    m = int(round(T * p))
    assert all(vals[i] >= vals[i + 1] for i in range(0, m))
    assert all(vals[i] <= vals[i + 1] for i in range(m, T))
    assert min(vals) >= 0.0


# --------------------------------------------------------------------------- #
# Interface: breach sequences vs counts, validation
# --------------------------------------------------------------------------- #
def test_breach_sequence_equals_counts():
    breaches = np.zeros(252, dtype=bool)
    breaches[:20] = True
    a = kupiec_pof_test(breaches, confidence=0.95)
    b = kupiec_pof_test(n_obs=252, n_breaches=20, confidence=0.95)
    assert a.as_dict() == b.as_dict()
    c = kupiec_pof_test(breaches.astype(int), confidence=0.95)   # 0/1 ints accepted
    assert c.n_breaches == 20


def test_p_value_and_verdict_consistent():
    for T, x in [(255, 7), (255, 21), (500, 30), (500, 12), (1000, 5)]:
        k = kupiec_pof_test(n_obs=T, n_breaches=x, confidence=0.95)
        assert k.p_value == pytest.approx(chi2.sf(k.lr_statistic, 1))
        assert k.passed == (k.lr_statistic < CHI2_5PCT)


def test_validation_errors():
    with pytest.raises(ValueError):
        kupiec_pof_test(n_obs=0, n_breaches=0)
    with pytest.raises(ValueError):
        kupiec_pof_test(n_obs=10, n_breaches=11)
    with pytest.raises(ValueError):
        kupiec_pof_test(np.array([0, 2, 1]))            # not 0/1
    with pytest.raises(ValueError):
        kupiec_pof_test(np.zeros(5, bool), n_obs=5)      # both forms
    with pytest.raises(ValueError):
        kupiec_pof_test(n_obs=10, n_breaches=1, confidence=1.0)
