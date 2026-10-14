import numpy as np
import pytest

from cse_risk_engine import GJRGARCHParams, fit_gjr_garch
from cse_risk_engine.simulation import simulate_gjr_garch_series

# "True" data-generating process used across tests.  Values are typical for
# daily equity log returns in percent (unconditional daily vol ~0.67 %).
TRUE_PARAMS = GJRGARCHParams(omega=0.02, alpha=0.05, gamma=0.10, beta=0.85, nu=6.0)


@pytest.fixture(scope="session")
def true_params():
    return TRUE_PARAMS


@pytest.fixture(scope="session")
def synthetic_returns():
    """5,000 synthetic percent log returns from TRUE_PARAMS (seed 2024)."""
    return simulate_gjr_garch_series(TRUE_PARAMS, n_obs=5000, seed=2024)


@pytest.fixture(scope="session")
def fitted(synthetic_returns):
    """Model fitted on the last 750 observations (the dissertation window)."""
    return fit_gjr_garch(synthetic_returns, window=750)


@pytest.fixture(scope="session")
def fitted_full(synthetic_returns):
    """Model fitted on all 5,000 observations (parameter-recovery test)."""
    return fit_gjr_garch(synthetic_returns, window=None)
