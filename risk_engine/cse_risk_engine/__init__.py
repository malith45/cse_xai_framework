"""
cse_risk_engine — Module M6 (Risk Engine) of the explainable ML decision-support
framework for the Colombo Stock Exchange.

Pure-function core with no dependency on Streamlit, any database, or any other
framework module.  Implements Chapter 4, Section 4.6.5:

    * GJR-GARCH(1,1) with Student-t innovations, fitted by ML on a 750-day
      rolling window (``volatility``)
    * Monte Carlo simulation of 10,000 return paths for 1- and 5-day horizons
      (``simulation``)
    * Value-at-Risk at 95 % / 99 % plus Expected Shortfall (``var``)
    * Stress testing: volatility multiplier, return shock, 2022-crisis
      calibration object (``stress``)
    * Kupiec (1995) proportion-of-failures backtest (``kupiec``) and the
      rolling out-of-sample VaR backtest that feeds it (``backtest``)

UNIT CONVENTION (applies to every public function in this package):
    All returns are DAILY LOG RETURNS EXPRESSED IN PERCENT.
    A value of 1.25 means a log return of +1.25 % ( ln(P_t / P_{t-1}) * 100 ).
    VaR and ES are reported as POSITIVE percent losses in the same log-return
    units, and as an LKR amount = notional * (1 - exp(-VaR_pct / 100)).
    Use :func:`log_returns_pct` to build the input series from prices.
"""

from .volatility import (
    GJRGARCHParams,
    GJRGARCHFit,
    ConvergenceError,
    fit_gjr_garch,
    filter_variance,
    next_variance,
    log_returns_pct,
    DEFAULT_WINDOW,
)
from .simulation import simulate_paths, SimulationResult, DEFAULT_N_PATHS
from .var import VaREstimate, VaRResult, value_at_risk, expected_shortfall, estimate_var
from .stress import (
    VolatilityMultiplier,
    ReturnShock,
    CrisisScenario,
    CrisisCalibration,
    StressResult,
    run_stress_test,
    calibrate_crisis,
)
from .kupiec import KupiecResult, kupiec_pof_test
from .backtest import BacktestResult, rolling_var_backtest

__all__ = [
    "GJRGARCHParams", "GJRGARCHFit", "ConvergenceError", "fit_gjr_garch",
    "filter_variance", "next_variance", "log_returns_pct", "DEFAULT_WINDOW",
    "simulate_paths", "SimulationResult", "DEFAULT_N_PATHS",
    "VaREstimate", "VaRResult", "value_at_risk", "expected_shortfall", "estimate_var",
    "VolatilityMultiplier", "ReturnShock", "CrisisScenario", "CrisisCalibration",
    "StressResult", "run_stress_test", "calibrate_crisis",
    "KupiecResult", "kupiec_pof_test",
    "BacktestResult", "rolling_var_backtest",
]

__version__ = "0.1.0"
