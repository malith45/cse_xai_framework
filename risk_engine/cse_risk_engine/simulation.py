"""
Monte Carlo simulation of GJR-GARCH(1,1)-t return paths (Chapter 4, §4.6.5).

The conditional-variance recursion (eq. 4.5) is propagated forward *path by
path* but *vectorised across paths* with NumPy, so a 10,000-path, 5-day
simulation completes in milliseconds.  Innovations are drawn from the fitted
standardised Student-t (unit variance):

    z ~ t(nu) * sqrt((nu - 2) / nu),     r_t = sigma_t * z_t

State at the start of the simulation is (r_T, sigma2_T) from the fit; the
first simulated day therefore has variance sigma2_{T+1} = eq. 4.5 applied to
(r_T, sigma2_T), which is deterministic.

UNITS: all returns in percent (log returns); variances in percent².
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .volatility import GJRGARCHFit, GJRGARCHParams, next_variance

DEFAULT_N_PATHS = 10_000


@dataclass(frozen=True)
class SimulationResult:
    """
    Attributes
    ----------
    returns : ndarray, shape (n_paths, horizon)
        Simulated daily log returns in percent.
    cumulative : ndarray, shape (n_paths,)
        h-day cumulative log return in percent (row sums of ``returns``).
    sigma2 : ndarray, shape (n_paths, horizon)
        Conditional variance used on each simulated day (percent²).
    horizon, n_paths, seed : simulation settings.
    initial_sigma2 : float
        sigma2_{T+1} actually used on day 1 (after any stress adjustment).
    initial_return : float
        r_T fed into the first recursion step (after any stress adjustment).
    """
    returns: np.ndarray
    cumulative: np.ndarray
    sigma2: np.ndarray
    horizon: int
    n_paths: int
    seed: Optional[int]
    initial_sigma2: float
    initial_return: float
    params: GJRGARCHParams


def standardised_t(rng: np.random.Generator, nu: float, size) -> np.ndarray:
    """Draw from a Student-t(nu) rescaled to unit variance (as ``arch`` does)."""
    if nu <= 2:
        raise ValueError("nu must be > 2")
    return rng.standard_t(nu, size=size) * np.sqrt((nu - 2.0) / nu)


def simulate_paths(model: Union[GJRGARCHFit, GJRGARCHParams],
                   horizon: int = 1,
                   n_paths: int = DEFAULT_N_PATHS,
                   seed: Optional[int] = None,
                   *,
                   initial_return: Optional[float] = None,
                   initial_sigma2: Optional[float] = None,
                   sigma2_multiplier: float = 1.0) -> SimulationResult:
    """Simulate ``n_paths`` return paths over ``horizon`` trading days.

    Parameters
    ----------
    model : GJRGARCHFit or GJRGARCHParams
        If a fit is given, its ``last_return`` and ``last_sigma2`` seed the
        recursion.  If bare parameters are given, ``initial_return`` and
        ``initial_sigma2`` must be supplied (crisis scenario / tests).
    horizon : int   Number of trading days (1 or 5 in the dissertation).
    n_paths : int   Number of paths (10,000 in the dissertation).
    seed : int, optional   Seed for ``numpy.random.default_rng``.
    initial_return : float, optional
        Override for r_T (percent).  Used by the return-shock stress test.
    initial_sigma2 : float, optional
        Override for sigma2_T (percent²).
    sigma2_multiplier : float
        Multiplier applied to the *first-day* conditional variance
        sigma2_{T+1} (so a volatility multiplier k maps to k²).  Used by the
        volatility-multiplier stress test.

    Returns
    -------
    SimulationResult
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if sigma2_multiplier <= 0:
        raise ValueError("sigma2_multiplier must be > 0")

    if isinstance(model, GJRGARCHFit):
        params = model.params
        r0 = model.last_return if initial_return is None else float(initial_return)
        s2_0 = model.last_sigma2 if initial_sigma2 is None else float(initial_sigma2)
    elif isinstance(model, GJRGARCHParams):
        params = model
        if initial_return is None or initial_sigma2 is None:
            raise ValueError("initial_return and initial_sigma2 are required when passing bare parameters")
        r0, s2_0 = float(initial_return), float(initial_sigma2)
    else:
        raise TypeError("model must be a GJRGARCHFit or GJRGARCHParams")
    if s2_0 <= 0:
        raise ValueError("initial_sigma2 must be > 0")

    rng = np.random.default_rng(seed)
    z = standardised_t(rng, params.nu, size=(n_paths, horizon))

    returns = np.empty((n_paths, horizon))
    sigma2 = np.empty((n_paths, horizon))

    # Day 1 variance is deterministic given (r_T, sigma2_T).
    s2 = np.full(n_paths, next_variance(params, r0, s2_0) * sigma2_multiplier)
    for t in range(horizon):
        sigma2[:, t] = s2
        returns[:, t] = np.sqrt(s2) * z[:, t]
        if t + 1 < horizon:
            s2 = next_variance(params, returns[:, t], s2)   # vectorised over paths

    return SimulationResult(
        returns=returns,
        cumulative=returns.sum(axis=1),
        sigma2=sigma2,
        horizon=horizon,
        n_paths=n_paths,
        seed=seed,
        initial_sigma2=float(sigma2[0, 0]),
        initial_return=r0,
        params=params,
    )


def simulate_gjr_garch_series(params: GJRGARCHParams, n_obs: int, seed: Optional[int] = None,
                              burn_in: int = 500) -> np.ndarray:
    """Generate a single long synthetic return series from known parameters
    (for testing parameter recovery).  Returns percent log returns, length n_obs."""
    rng = np.random.default_rng(seed)
    total = n_obs + burn_in
    z = standardised_t(rng, params.nu, size=total)
    r = np.empty(total)
    s2 = params.unconditional_variance if params.is_stationary else params.omega
    r_prev = 0.0
    for t in range(total):
        if t > 0:
            s2 = next_variance(params, r_prev, s2)
        r[t] = np.sqrt(s2) * z[t]
        r_prev = r[t]
    return r[burn_in:]
