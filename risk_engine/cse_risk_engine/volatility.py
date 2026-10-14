"""
GJR-GARCH(1,1)-t volatility model (Chapter 4, eq. 4.5).

    sigma2_t = omega + (alpha + gamma * I_{t-1}) * eps2_{t-1} + beta * sigma2_{t-1}
    I_{t-1}  = 1 if the previous return was negative, else 0
    eps_t / sigma_t ~ standardised Student-t(nu)   (unit variance)

Mean equation: ZERO.  Chapter 4 specifies no mean equation and conditions the
leverage indicator on the *prior return* while writing the recursion in eps²;
both hold together only when eps_t = r_t, i.e. a zero-mean model.  Estimation
uses the ``arch`` library (``mean='Zero', vol='GARCH', p=1, o=1, q=1, dist='t'``),
whose GJR parametrisation is exactly the recursion above.

UNITS: inputs are daily log returns in PERCENT.  ``omega`` is therefore in
percent² and ``sigma2`` values are percent².
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np

DEFAULT_WINDOW = 750  # trading days (Chapter 4, §4.6.5)

_PARAM_ORDER = ("omega", "alpha", "gamma", "beta", "nu")
_ARCH_NAMES = ("omega", "alpha[1]", "gamma[1]", "beta[1]", "nu")


class ConvergenceError(RuntimeError):
    """Raised when maximum-likelihood estimation does not converge or yields
    parameters that fail sanity checks.  The engine never returns silently
    from a failed fit."""


def log_returns_pct(prices: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert a price series to daily log returns in percent.

    r_t = 100 * ln(P_t / P_{t-1}).  Output has length len(prices) - 1.
    """
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("prices must be a 1-D sequence with at least two values")
    if np.any(p <= 0):
        raise ValueError("prices must be strictly positive")
    return 100.0 * np.diff(np.log(p))


@dataclass(frozen=True)
class GJRGARCHParams:
    """Parameters of the GJR-GARCH(1,1)-t model.  ``omega`` in percent²;
    ``alpha``, ``gamma``, ``beta`` dimensionless; ``nu`` degrees of freedom (> 2)."""
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float

    def __post_init__(self):
        if self.omega <= 0:
            raise ValueError("omega must be > 0")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be >= 0")
        if self.alpha + self.gamma < 0:
            raise ValueError("alpha + gamma must be >= 0 (variance would be negative after a negative shock)")
        if self.nu <= 2:
            raise ValueError("nu must be > 2 for the Student-t to have finite variance")

    @property
    def persistence(self) -> float:
        """alpha + gamma/2 + beta (assumes symmetric innovations, so P(eps<0)=1/2)."""
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def is_stationary(self) -> bool:
        return self.persistence < 1.0

    @property
    def unconditional_variance(self) -> float:
        """Long-run variance omega / (1 - persistence), percent².  inf if non-stationary."""
        return self.omega / (1.0 - self.persistence) if self.is_stationary else float("inf")

    def as_array(self) -> np.ndarray:
        return np.array([self.omega, self.alpha, self.gamma, self.beta, self.nu], dtype=float)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GJRGARCHFit:
    """Result of :func:`fit_gjr_garch`.

    Attributes
    ----------
    params : GJRGARCHParams
    std_errors : dict[str, float]
        Robust (sandwich) standard errors keyed by parameter name.
    log_likelihood : float
    converged : bool
        Always True for a returned object (failures raise ``ConvergenceError``);
        kept explicit so downstream code never has to guess.
    n_obs : int
        Number of returns used in the fit (== window length).
    window : tuple[int, int]
        Half-open index range ``[start, end)`` of the input array used.
    last_return : float
        r_T, the last in-sample return (percent).  Equals eps_T under zero mean.
    last_sigma2 : float
        sigma2_T, the last in-sample conditional variance (percent²).
    warnings : tuple[str, ...]
        Non-fatal diagnostics (e.g. persistence close to 1).
    """
    params: GJRGARCHParams
    std_errors: dict
    log_likelihood: float
    converged: bool
    n_obs: int
    window: tuple
    last_return: float
    last_sigma2: float
    warnings: tuple = field(default_factory=tuple)
    aic: Optional[float] = None
    bic: Optional[float] = None

    @property
    def next_sigma2(self) -> float:
        """sigma2_{T+1}: the (deterministic) one-step-ahead conditional variance."""
        return next_variance(self.params, self.last_return, self.last_sigma2)

    @property
    def next_sigma(self) -> float:
        return float(np.sqrt(self.next_sigma2))

    def summary(self) -> str:
        p = self.params
        lines = [
            "GJR-GARCH(1,1)-t  (zero mean, log returns in %)",
            f"  n_obs={self.n_obs}  window={self.window}  logL={self.log_likelihood:.3f}  converged={self.converged}",
        ]
        for k in _PARAM_ORDER:
            lines.append(f"  {k:6s} = {getattr(p, k):10.6f}   (s.e. {self.std_errors.get(k, float('nan')):.6f})")
        lines.append(f"  persistence={p.persistence:.4f}  uncond. daily vol={np.sqrt(p.unconditional_variance):.4f}%")
        lines.append(f"  last_return={self.last_return:.4f}%  last_sigma={np.sqrt(self.last_sigma2):.4f}%  next_sigma={self.next_sigma:.4f}%")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def next_variance(params: GJRGARCHParams, prev_return: float | np.ndarray,
                  prev_sigma2: float | np.ndarray) -> float | np.ndarray:
    """One step of eq. 4.5.  Vectorised: ``prev_return`` and ``prev_sigma2``
    may be arrays of equal shape (one entry per Monte Carlo path).

    sigma2_t = omega + (alpha + gamma * 1[r_{t-1} < 0]) * r_{t-1}² + beta * sigma2_{t-1}
    """
    r = np.asarray(prev_return, dtype=float)
    s2 = np.asarray(prev_sigma2, dtype=float)
    indicator = (r < 0.0).astype(float)
    out = params.omega + (params.alpha + params.gamma * indicator) * r ** 2 + params.beta * s2
    return float(out) if out.ndim == 0 else out


def filter_variance(params: GJRGARCHParams, returns_pct: Sequence[float] | np.ndarray,
                    sigma2_0: Optional[float] = None) -> np.ndarray:
    """Run the variance recursion through an observed return series (no
    estimation).  Returns sigma2_t for t = 0..n-1, percent².

    sigma2_0 defaults to the unconditional variance (or the sample variance if
    the process is non-stationary).  Used by the rolling backtest to update
    the conditional variance between refits.
    """
    r = np.asarray(returns_pct, dtype=float)
    n = r.size
    s2 = np.empty(n)
    if sigma2_0 is None:
        sigma2_0 = params.unconditional_variance if params.is_stationary else float(np.var(r))
    s2[0] = sigma2_0
    for t in range(1, n):
        s2[t] = next_variance(params, r[t - 1], s2[t - 1])
    return s2


def _sanity_check(params: GJRGARCHParams, std_errors: dict) -> list[str]:
    msgs = []
    if not params.is_stationary:
        msgs.append(f"non-stationary: persistence={params.persistence:.4f} >= 1; unconditional variance undefined")
    elif params.persistence > 0.995:
        msgs.append(f"persistence={params.persistence:.4f} is very close to 1 (near-IGARCH)")
    if params.nu > 200:
        msgs.append(f"nu={params.nu:.1f}: innovations are effectively Gaussian")
    for k, v in std_errors.items():
        if not np.isfinite(v):
            msgs.append(f"standard error of {k} is not finite (Hessian may be singular)")
    return msgs


def fit_gjr_garch(returns_pct: Sequence[float] | np.ndarray,
                  window: Optional[int] = DEFAULT_WINDOW,
                  *,
                  end: Optional[int] = None,
                  max_attempts: int = 3,
                  maxiter: int = 1000,
                  strict: bool = True) -> GJRGARCHFit:
    """Fit GJR-GARCH(1,1) with Student-t innovations by maximum likelihood.

    Parameters
    ----------
    returns_pct : 1-D array of daily log returns in PERCENT.
    window : int or None
        Number of most recent observations to use (default 750).  ``None``
        uses the whole series (useful in tests with synthetic data).
    end : int, optional
        Exclusive end index into ``returns_pct``; the fit uses
        ``returns_pct[end-window:end]``.  Defaults to ``len(returns_pct)``.
        This is how the rolling backtest walks forward.
    max_attempts : int
        Number of optimiser attempts with alternative starting values before
        raising ``ConvergenceError``.
    maxiter : int
        Optimiser iteration cap per attempt.
    strict : bool
        If True (default) a non-stationary fit (persistence >= 1) is treated
        as a failure.  If False it is returned with a warning.

    Returns
    -------
    GJRGARCHFit

    Raises
    ------
    ValueError   on bad inputs (too few observations, NaNs, zero variance).
    ConvergenceError   if the optimiser does not converge on any attempt or the
        parameters fail sanity checks.  The exception message carries the
        arch convergence flag and the offending parameter values.
    """
    from arch import arch_model  # imported lazily so the module imports fast

    r_all = np.asarray(returns_pct, dtype=float)
    if r_all.ndim != 1:
        raise ValueError("returns_pct must be 1-D")
    n_all = r_all.size
    end = n_all if end is None else int(end)
    if end < 1 or end > n_all:
        raise ValueError(f"end={end} out of range for series of length {n_all}")
    start = 0 if window is None else max(0, end - int(window))
    if window is not None and end - start < window:
        raise ValueError(f"need {window} observations for the window but only {end - start} available")
    r = r_all[start:end]
    if r.size < 100:
        raise ValueError(f"at least 100 observations are needed to fit GJR-GARCH-t; got {r.size}")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns_pct contains NaN or inf")
    if np.std(r) == 0.0:
        raise ValueError("returns_pct has zero variance; cannot fit a volatility model")
    if np.std(r) > 30 or np.std(r) < 0.05:
        warnings.warn(
            f"daily return std = {np.std(r):.4g}; expected order 0.1–10 for daily percent log returns. "
            "Check that inputs are in PERCENT, not decimals.", stacklevel=2)

    am = arch_model(r, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)

    # Alternative starting values: (omega, alpha, gamma, beta, nu) in arch order.
    v = float(np.var(r))
    starting = [
        None,                                            # arch defaults
        np.array([0.05 * v, 0.03, 0.10, 0.85, 6.0]),
        np.array([0.10 * v, 0.10, 0.05, 0.75, 8.0]),
        np.array([0.02 * v, 0.02, 0.15, 0.90, 5.0]),
    ]
    last_error: Optional[str] = None
    for attempt in range(max_attempts):
        sv = starting[attempt % len(starting)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(disp="off", starting_values=sv, options={"maxiter": maxiter},
                             show_warning=False)
        except Exception as exc:  # optimiser / linear algebra failure
            last_error = f"attempt {attempt + 1}: optimiser raised {type(exc).__name__}: {exc}"
            continue
        if getattr(res, "convergence_flag", 1) != 0:
            last_error = f"attempt {attempt + 1}: convergence_flag={res.convergence_flag}"
            continue
        vals = [float(res.params[k]) for k in _ARCH_NAMES]
        try:
            params = GJRGARCHParams(*vals)
        except ValueError as exc:
            last_error = f"attempt {attempt + 1}: invalid parameters {dict(zip(_PARAM_ORDER, vals))}: {exc}"
            continue
        std_errors = {k: float(res.std_err[a]) for k, a in zip(_PARAM_ORDER, _ARCH_NAMES)}
        diags = _sanity_check(params, std_errors)
        if strict and not params.is_stationary:
            last_error = f"attempt {attempt + 1}: {diags[0]}"
            continue
        cond_var = np.asarray(res.conditional_volatility, dtype=float) ** 2
        return GJRGARCHFit(
            params=params,
            std_errors=std_errors,
            log_likelihood=float(res.loglikelihood),
            converged=True,
            n_obs=int(r.size),
            window=(start, end),
            last_return=float(r[-1]),
            last_sigma2=float(cond_var[-1]),
            warnings=tuple(diags),
            aic=float(res.aic),
            bic=float(res.bic),
        )
    raise ConvergenceError(
        f"GJR-GARCH(1,1)-t estimation failed after {max_attempts} attempt(s) on "
        f"{r.size} observations [{start}:{end}). Last error: {last_error}")
