"""
Lightweight Monte Carlo stress engine.

Runs in the hosted app from *stored* GJR-GARCH(1,1)-t parameters supplied by
the repository. It performs no estimation and reads no raw data; it is the only
computation the dashboard is allowed to do online (Chapter 4, offline-training /
online-serving pattern).
"""
from __future__ import annotations

import numpy as np

SEED = 7


def simulate_returns(params: dict, horizon_days: int, n_paths: int = 10_000,
                     vol_multiplier: float = 1.0, initial_shock: float = 0.0) -> np.ndarray:
    """Return an array of simulated cumulative log returns over `horizon_days`."""
    rng = np.random.default_rng(SEED)
    mu, omega = params["mu"], params["omega"]
    alpha, gamma, beta, nu = params["alpha"], params["gamma"], params["beta"], params["nu"]

    sigma2 = np.full(n_paths, (params["sigma_last"] * vol_multiplier) ** 2)
    eps_prev = np.full(n_paths, params["ret_last"] - mu + initial_shock)
    cum = np.zeros(n_paths)

    # Student-t innovations scaled to unit variance.
    scale = np.sqrt((nu - 2) / nu)
    for _ in range(horizon_days):
        neg = (eps_prev < 0).astype(float)
        sigma2 = omega + (alpha + gamma * neg) * eps_prev**2 + beta * sigma2
        z = rng.standard_t(nu, n_paths) * scale
        eps = np.sqrt(sigma2) * z
        cum += mu + eps
        eps_prev = eps
    return cum


def value_at_risk(returns: np.ndarray, confidence: float) -> float:
    """VaR as a positive loss fraction: negative of the (1-c) quantile."""
    return float(-np.quantile(returns, 1 - confidence))
