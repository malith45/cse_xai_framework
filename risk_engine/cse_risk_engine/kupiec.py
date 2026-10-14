"""
Kupiec (1995) proportion-of-failures (POF) test — Chapter 4, eq. 4.6.

    LR_POF = -2 ln[(1-p)^(T-x) p^x] + 2 ln[(1-x/T)^(T-x) (x/T)^x]  ~  chi²(1)

where T = number of out-of-sample observations, x = number of VaR breaches and
p = 1 - confidence (expected breach probability, e.g. 0.05 for 95 % VaR).

Edge cases (the raw formula takes ln 0):
    x = 0 : the second bracket is (1-0)^T * 0^0 = 1 (0^0 := 1), so LR = -2 T ln(1-p)
    x = T : the second bracket is 0^0 * 1^T = 1,             so LR = -2 T ln p
Both are the exact limits of the general formula and are handled explicitly.

Verdict: "pass" (do not reject the VaR model) when the p-value >= significance
(default 5 %, i.e. LR < 3.841).  Note the test is two-sided in spirit: too few
breaches (an over-conservative model) also fails.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class KupiecResult:
    n_obs: int              # T
    n_breaches: int         # x
    observed_rate: float    # x / T
    expected_rate: float    # p
    lr_statistic: float     # LR_POF
    p_value: float
    significance: float
    critical_value: float   # chi²(1) critical value at `significance`
    passed: bool            # True = VaR model NOT rejected

    @property
    def verdict(self) -> str:
        return "PASS (not rejected)" if self.passed else "FAIL (rejected)"

    def as_dict(self) -> dict:
        return {
            "T": self.n_obs, "x": self.n_breaches,
            "observed_rate": self.observed_rate, "expected_rate": self.expected_rate,
            "LR_POF": self.lr_statistic, "p_value": self.p_value,
            "significance": self.significance, "critical_value": self.critical_value,
            "passed": self.passed, "verdict": self.verdict,
        }

    def __str__(self) -> str:
        return (f"Kupiec POF: T={self.n_obs}, x={self.n_breaches}, observed={self.observed_rate:.4%}, "
                f"expected={self.expected_rate:.2%}, LR={self.lr_statistic:.4f}, p={self.p_value:.4f} "
                f"-> {self.verdict} at {self.significance:.0%}")


def kupiec_lr_statistic(n_obs: int, n_breaches: int, p: float) -> float:
    """LR_POF with the x = 0 and x = T limits handled exactly."""
    T, x = int(n_obs), int(n_breaches)
    if T <= 0:
        raise ValueError("n_obs must be > 0")
    if not 0 <= x <= T:
        raise ValueError("n_breaches must satisfy 0 <= x <= T")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if x == 0:
        return float(-2.0 * T * np.log(1.0 - p))
    if x == T:
        return float(-2.0 * T * np.log(p))
    pi = x / T
    ll_null = (T - x) * np.log(1.0 - p) + x * np.log(p)
    ll_alt = (T - x) * np.log(1.0 - pi) + x * np.log(pi)
    lr = -2.0 * ll_null + 2.0 * ll_alt
    return float(max(lr, 0.0))  # numerical guard: LR is a ratio of nested likelihoods, so >= 0


def kupiec_pof_test(breaches: Optional[Sequence[bool] | np.ndarray] = None,
                    *,
                    n_obs: Optional[int] = None,
                    n_breaches: Optional[int] = None,
                    confidence: float = 0.95,
                    significance: float = 0.05) -> KupiecResult:
    """Kupiec proportion-of-failures test.

    Supply EITHER ``breaches`` (a boolean/0-1 sequence, one entry per
    out-of-sample day, True where the realised loss exceeded VaR) OR the
    counts ``n_obs`` and ``n_breaches``.

    ``confidence`` is the VaR confidence level (0.95 or 0.99); the expected
    breach probability is p = 1 - confidence.  ``significance`` is the test
    size (default 0.05).
    """
    if breaches is not None:
        if n_obs is not None or n_breaches is not None:
            raise ValueError("pass breaches OR (n_obs, n_breaches), not both")
        b = np.asarray(breaches)
        if b.ndim != 1:
            raise ValueError("breaches must be 1-D")
        if b.dtype != bool and not np.all(np.isin(b, [0, 1])):
            raise ValueError("breaches must be boolean or 0/1")
        T, x = int(b.size), int(b.astype(bool).sum())
    else:
        if n_obs is None or n_breaches is None:
            raise ValueError("supply breaches or both n_obs and n_breaches")
        T, x = int(n_obs), int(n_breaches)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if not 0.0 < significance < 1.0:
        raise ValueError("significance must be in (0, 1)")

    p = 1.0 - confidence
    lr = kupiec_lr_statistic(T, x, p)
    p_value = float(chi2.sf(lr, df=1))
    crit = float(chi2.ppf(1.0 - significance, df=1))
    return KupiecResult(
        n_obs=T, n_breaches=x, observed_rate=x / T, expected_rate=p,
        lr_statistic=lr, p_value=p_value, significance=significance,
        critical_value=crit, passed=bool(p_value >= significance),
    )
