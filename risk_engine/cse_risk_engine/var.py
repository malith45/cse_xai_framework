"""
Value-at-Risk and Expected Shortfall from simulated h-day returns
(Chapter 4, §4.6.5).

    VaR_c  = -(the (1-c) quantile of the simulated h-day return distribution)
             e.g. 95 % VaR = -(5th percentile), 99 % VaR = -(1st percentile)
    ES_c   = -(mean of simulated returns at or below that quantile)
             i.e. the mean loss beyond VaR

UNITS
    var_pct / es_pct : positive percent losses in LOG-RETURN units
                       (same units as the simulated returns).
    var_lkr / es_lkr : LKR loss on a user-supplied notional using the exact
                       log-return conversion
                           loss = notional * (1 - exp(-var_pct / 100)).
                       Chapter 4 step (6) defines returns as log returns, so
                       this is the only conversion consistent with the text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from .simulation import simulate_paths, SimulationResult, DEFAULT_N_PATHS
from .volatility import GJRGARCHFit, GJRGARCHParams

DEFAULT_CONFIDENCE = (0.95, 0.99)
DEFAULT_HORIZONS = (1, 5)


def log_loss_to_lkr(var_pct: float, notional: float) -> float:
    """Exact LKR loss for a log-return loss of ``var_pct`` percent."""
    return float(notional * (1.0 - np.exp(-var_pct / 100.0)))


@dataclass(frozen=True)
class VaREstimate:
    horizon: int
    confidence: float
    var_pct: float
    var_lkr: float
    es_pct: float
    es_lkr: float
    notional: float
    n_paths: int

    def as_dict(self) -> dict:
        return {
            "horizon_days": self.horizon, "confidence": self.confidence,
            "var_pct": self.var_pct, "var_lkr": self.var_lkr,
            "es_pct": self.es_pct, "es_lkr": self.es_lkr,
            "notional_lkr": self.notional, "n_paths": self.n_paths,
        }


@dataclass(frozen=True)
class VaRResult:
    """All VaR/ES estimates for one model, keyed by (horizon, confidence)."""
    estimates: dict
    notional: float
    seed: Optional[int]
    n_paths: int

    def get(self, horizon: int, confidence: float) -> VaREstimate:
        return self.estimates[(horizon, confidence)]

    def table(self) -> list[dict]:
        return [e.as_dict() for e in self.estimates.values()]

    def __str__(self) -> str:
        rows = [f"VaR / ES on notional LKR {self.notional:,.0f}  (n_paths={self.n_paths}, seed={self.seed})",
                f"{'h':>3} {'conf':>6} {'VaR %':>9} {'VaR LKR':>14} {'ES %':>9} {'ES LKR':>14}"]
        for e in self.estimates.values():
            rows.append(f"{e.horizon:>3} {e.confidence:>6.2%} {e.var_pct:>9.4f} {e.var_lkr:>14,.0f} "
                        f"{e.es_pct:>9.4f} {e.es_lkr:>14,.0f}")
        return "\n".join(rows)


def value_at_risk(cumulative_returns_pct: np.ndarray, confidence: float) -> float:
    """VaR (positive percent, log-return units) = -(1-confidence) quantile.

    Uses the empirical (linear-interpolation) quantile of the simulated
    distribution, i.e. ``numpy.percentile`` with the default method.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    x = np.asarray(cumulative_returns_pct, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("cumulative_returns_pct must be a non-empty 1-D array")
    return float(-np.percentile(x, 100.0 * (1.0 - confidence)))


def expected_shortfall(cumulative_returns_pct: np.ndarray, confidence: float) -> float:
    """ES (positive percent) = mean loss over paths whose return <= -VaR."""
    x = np.asarray(cumulative_returns_pct, dtype=float)
    var = value_at_risk(x, confidence)
    tail = x[x <= -var]
    if tail.size == 0:  # cannot happen with interpolated quantile, but guard anyway
        return var
    return float(-tail.mean())


def var_from_simulation(sim: SimulationResult, confidence: float, notional: float) -> VaREstimate:
    var_pct = value_at_risk(sim.cumulative, confidence)
    es_pct = expected_shortfall(sim.cumulative, confidence)
    return VaREstimate(
        horizon=sim.horizon, confidence=confidence,
        var_pct=var_pct, var_lkr=log_loss_to_lkr(var_pct, notional),
        es_pct=es_pct, es_lkr=log_loss_to_lkr(es_pct, notional),
        notional=notional, n_paths=sim.n_paths,
    )


def estimate_var(model: Union[GJRGARCHFit, GJRGARCHParams],
                 notional: float,
                 horizons: Sequence[int] = DEFAULT_HORIZONS,
                 confidence: Sequence[float] = DEFAULT_CONFIDENCE,
                 n_paths: int = DEFAULT_N_PATHS,
                 seed: Optional[int] = None,
                 **sim_kwargs) -> VaRResult:
    """Monte Carlo VaR and ES for every (horizon, confidence) pair.

    Parameters
    ----------
    model : GJRGARCHFit (normal use) or GJRGARCHParams (with initial state in ``sim_kwargs``).
    notional : float   Position value in LKR.
    horizons : e.g. (1, 5).   confidence : e.g. (0.95, 0.99).
    n_paths, seed : simulation settings.  One simulation per horizon; the
        same seed is reused for every horizon so results are reproducible.
    **sim_kwargs : forwarded to :func:`simulate_paths` (stress overrides).
    """
    if notional < 0:
        raise ValueError("notional must be >= 0")
    estimates = {}
    for h in horizons:
        sim = simulate_paths(model, horizon=int(h), n_paths=n_paths, seed=seed, **sim_kwargs)
        for c in confidence:
            estimates[(int(h), float(c))] = var_from_simulation(sim, float(c), float(notional))
    return VaRResult(estimates=estimates, notional=float(notional), seed=seed, n_paths=n_paths)
