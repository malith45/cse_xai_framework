"""End-to-end demonstration of cse_risk_engine on synthetic data.

Run:  python example.py
Swap ``returns`` for ``log_returns_pct(close_prices)`` once CSE data arrive.
"""
import numpy as np

from cse_risk_engine import (
    GJRGARCHParams, fit_gjr_garch, estimate_var, run_stress_test, rolling_var_backtest,
    VolatilityMultiplier, ReturnShock, CrisisScenario, CrisisCalibration, calibrate_crisis,
)
from cse_risk_engine.simulation import simulate_gjr_garch_series

# 1. Data (synthetic stand-in for a S&P SL20 constituent's percent log returns)
true = GJRGARCHParams(omega=0.02, alpha=0.05, gamma=0.10, beta=0.85, nu=6.0)
returns = simulate_gjr_garch_series(true, n_obs=1500, seed=2026)

# 2. Volatility model: 750-day rolling window, most recent window
fit = fit_gjr_garch(returns, window=750)
print(fit.summary(), "\n")

# 3. Monte Carlo VaR / ES on a LKR 1,000,000 position
notional = 1_000_000
base = estimate_var(fit, notional, horizons=(1, 5), confidence=(0.95, 0.99), n_paths=10_000, seed=42)
print(base, "\n")

# 4. Stress tests (no refitting)
print(run_stress_test(fit, VolatilityMultiplier(2.0), notional, seed=42, baseline=base), "\n")
print(run_stress_test(fit, ReturnShock(-5.0), notional, seed=42, baseline=base), "\n")

# 4c. Crisis scenario — placeholder until the 2022 CSE window is available ...
placeholder = CrisisCalibration.placeholder(fit)
print(run_stress_test(fit, CrisisScenario(placeholder), notional, seed=42, baseline=base), "\n")
# ... then replace with a real calibration:
#   crisis = calibrate_crisis(log_returns_pct(aspi_close["2022-01-03":"2022-12-30"]),
#                             label="CSE 2022 sovereign-default crisis",
#                             period=("2022-01-03", "2022-12-30"), source="CSE Data Library")

# 5. Kupiec backtest on rolling out-of-sample 1-day VaR
bt = rolling_var_backtest(returns, window=750, confidence=(0.95, 0.99), refit_every=21, n_paths=10_000, seed=0)
print(bt)
