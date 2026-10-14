"""
Rolling out-of-sample 1-day VaR backtest feeding the Kupiec test
(Chapter 4, §4.6.5: "applied to rolling out-of-sample 1-day VaR at 95 % and 99 %").

For each out-of-sample day t (t = window, ..., n-1):
    1. the model is (re)fitted on returns[t-window : t]  — strictly before day t;
    2. VaR_t is estimated by Monte Carlo from the fitted state (r_{t-1}, sigma2_{t-1});
    3. a breach is recorded if r_t < -VaR_t.

``refit_every`` controls how often step 1 re-estimates the parameters.  The
dissertation text implies daily refitting (``refit_every=1``, the default).
Between refits the conditional variance is still updated every day through
eq. 4.5 with the last fitted parameters (no information is stale except the
parameter values themselves).

UNITS: percent log returns throughout.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .volatility import fit_gjr_garch, next_variance, ConvergenceError, DEFAULT_WINDOW, GJRGARCHFit
from .simulation import simulate_paths, DEFAULT_N_PATHS
from .var import value_at_risk, DEFAULT_CONFIDENCE
from .kupiec import kupiec_pof_test, KupiecResult


@dataclass(frozen=True)
class BacktestResult:
    """
    Attributes
    ----------
    index : ndarray[int]         positions of the out-of-sample days in the input series
    realised : ndarray[float]    realised returns r_t (percent)
    var : dict[float, ndarray]   VaR_t per confidence level (positive percent)
    breaches : dict[float, ndarray[bool]]
    kupiec : dict[float, KupiecResult]
    refit_index : ndarray[int]   days on which the model was (re)fitted
    failed_fits : tuple[int]     days where estimation failed (only if on_fit_failure="carry")
    window, horizon, n_paths, seed, refit_every : settings
    """
    index: np.ndarray
    realised: np.ndarray
    var: dict
    breaches: dict
    kupiec: dict
    refit_index: np.ndarray
    failed_fits: tuple
    window: int
    horizon: int
    n_paths: int
    seed: Optional[int]
    refit_every: int

    def __str__(self) -> str:
        lines = [f"Rolling VaR backtest: window={self.window}, horizon={self.horizon}, "
                 f"out-of-sample days={self.index.size}, refits={self.refit_index.size}, "
                 f"failed fits={len(self.failed_fits)}"]
        for c, k in self.kupiec.items():
            lines.append(f"  {c:.0%}: " + str(k))
        return "\n".join(lines)


def rolling_var_backtest(returns_pct: Sequence[float] | np.ndarray,
                         window: int = DEFAULT_WINDOW,
                         confidence: Sequence[float] = DEFAULT_CONFIDENCE,
                         *,
                         horizon: int = 1,
                         refit_every: int = 1,
                         n_paths: int = DEFAULT_N_PATHS,
                         seed: Optional[int] = None,
                         start: Optional[int] = None,
                         end: Optional[int] = None,
                         significance: float = 0.05,
                         on_fit_failure: str = "raise") -> BacktestResult:
    """Walk the series forward producing out-of-sample VaR and Kupiec results.

    Parameters
    ----------
    returns_pct : percent log returns, length n.
    window : estimation window (750).
    confidence : VaR confidence levels to backtest.
    horizon : 1 for the Kupiec test in the dissertation.  If > 1, the
        realised h-day return is compared with the h-day VaR on
        non-overlapping blocks (t, t+h, ...).
    refit_every : re-estimate parameters every k out-of-sample days.
    n_paths, seed : Monte Carlo settings; the per-day seed is ``seed + t``.
    start, end : optional bounds on the out-of-sample day index t
        (default: window .. n-1).  Useful for testing on a slice.
    on_fit_failure : "raise" (default) propagates ``ConvergenceError``;
        "carry" keeps the previous parameters, records the day in
        ``failed_fits`` and emits a warning.  There is no silent option.
    """
    r = np.asarray(returns_pct, dtype=float)
    n = r.size
    if on_fit_failure not in ("raise", "carry"):
        raise ValueError('on_fit_failure must be "raise" or "carry"')
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    t0 = window if start is None else max(int(start), window)
    t1 = n if end is None else min(int(end), n)
    if t1 - t0 < horizon:
        raise ValueError(f"no out-of-sample observations: need n > window (+horizon); n={n}, window={window}")

    conf = [float(c) for c in confidence]
    days, realised = [], []
    var_series = {c: [] for c in conf}
    refit_days, failed = [], []

    fit: Optional[GJRGARCHFit] = None
    params = None
    r_prev = None      # r_{t-1}
    s2_prev = None     # sigma2_{t-1}

    step = 0
    t = t0
    while t + horizon - 1 < t1:
        need_fit = fit is None or (step % refit_every == 0)
        if need_fit:
            try:
                fit = fit_gjr_garch(r, window=window, end=t)
                params, r_prev, s2_prev = fit.params, fit.last_return, fit.last_sigma2
                refit_days.append(t)
            except ConvergenceError as exc:
                if on_fit_failure == "raise" or fit is None:
                    raise ConvergenceError(f"rolling backtest failed at t={t}: {exc}") from exc
                failed.append(t)
                warnings.warn(f"fit failed at t={t}; carrying previous parameters", stacklevel=2)
                # fall through with the carried state (already advanced below)
        day_seed = None if seed is None else int(seed) + t
        sim = simulate_paths(params, horizon=horizon, n_paths=n_paths, seed=day_seed,
                             initial_return=r_prev, initial_sigma2=s2_prev)
        real = float(r[t:t + horizon].sum())
        for c in conf:
            var_series[c].append(value_at_risk(sim.cumulative, c))
        days.append(t)
        realised.append(real)

        # advance the state through the realised returns (eq. 4.5)
        for k in range(horizon):
            s2_prev = next_variance(params, r_prev, s2_prev)
            r_prev = float(r[t + k])
        t += horizon
        step += 1

    index = np.asarray(days, dtype=int)
    realised_arr = np.asarray(realised)
    var_out = {c: np.asarray(v) for c, v in var_series.items()}
    breaches = {c: realised_arr < -var_out[c] for c in conf}
    kup = {c: kupiec_pof_test(breaches[c], confidence=c, significance=significance) for c in conf}
    return BacktestResult(index=index, realised=realised_arr, var=var_out, breaches=breaches,
                         kupiec=kup, refit_index=np.asarray(refit_days, dtype=int),
                         failed_fits=tuple(failed), window=window, horizon=horizon,
                         n_paths=n_paths, seed=seed, refit_every=refit_every)
