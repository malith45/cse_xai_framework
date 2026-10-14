"""
Stress testing (Chapter 4, FR13 and §4.6.5).  All three scenarios reuse the
fitted model WITHOUT refitting:

(a) VolatilityMultiplier(factor)
        The first-day conditional volatility sigma_{T+1} is scaled by ``factor``
        (variance by factor²); the recursion then runs forward normally.

(b) ReturnShock(shock_pct)
        An initial return of ``shock_pct`` (percent, normally negative) is
        imposed as r_T — the return "just realised" before the horizon starts.
        It raises sigma2_{T+1} through the (alpha + gamma) term and the
        simulation runs forward from there.  The shock itself is NOT counted
        in the h-day return: the result is the conditional VaR *after* the
        shock has happened.

(c) CrisisScenario(calibration)
        The GJR-GARCH parameters are replaced by a ``CrisisCalibration``
        object estimated from a crisis period (the 2022 crisis once CSE data
        arrive) via :func:`calibrate_crisis`.  By default the simulation starts
        from the crisis model's unconditional variance (i.e. "the market
        behaves like the crisis period from today"); pass
        ``use_current_state=True`` to start from the current fit's state
        instead.  No crisis numbers are hard-coded: the only built-in object,
        ``CrisisCalibration.placeholder(fit)``, reuses the current fit's own
        parameters and is flagged ``is_placeholder=True`` everywhere it appears.

UNITS: percent log returns; LKR via notional * (1 - exp(-VaR/100)).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

from .volatility import GJRGARCHFit, GJRGARCHParams, fit_gjr_garch
from .var import VaRResult, estimate_var, DEFAULT_CONFIDENCE, DEFAULT_HORIZONS
from .simulation import DEFAULT_N_PATHS

PLACEHOLDER_LABEL = "PLACEHOLDER — NOT CALIBRATED TO THE 2022 CRISIS (uses current fit parameters)"


# --------------------------------------------------------------------------- #
# Crisis calibration interface
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CrisisCalibration:
    """Swappable calibration object for the crisis scenario.

    Attributes
    ----------
    params : GJRGARCHParams   fitted on the crisis window (percent units)
    label : str               e.g. "CSE 2022 sovereign-default crisis"
    period : tuple[str, str]  ISO dates of the calibration window (informational)
    source : str              data source description (informational)
    initial_sigma2 : float, optional
        Starting variance for the scenario (percent²).  Defaults to the crisis
        model's unconditional variance when None.
    is_placeholder : bool     True only for the built-in placeholder.
    log_likelihood, n_obs : provenance of the calibration fit.
    """
    params: GJRGARCHParams
    label: str
    period: tuple = ("", "")
    source: str = ""
    initial_sigma2: Optional[float] = None
    is_placeholder: bool = False
    log_likelihood: Optional[float] = None
    n_obs: Optional[int] = None

    @classmethod
    def placeholder(cls, fit: GJRGARCHFit) -> "CrisisCalibration":
        """Clearly-labelled stand-in until real crisis data are available.
        Reuses ``fit.params``; carries no invented numbers."""
        return cls(params=fit.params, label=PLACEHOLDER_LABEL, is_placeholder=True,
                   log_likelihood=fit.log_likelihood, n_obs=fit.n_obs,
                   source="current model fit (placeholder)")

    @property
    def start_sigma2(self) -> float:
        if self.initial_sigma2 is not None:
            return float(self.initial_sigma2)
        if not self.params.is_stationary:
            raise ValueError("crisis parameters are non-stationary; supply initial_sigma2 explicitly")
        return self.params.unconditional_variance


def calibrate_crisis(returns_pct: Sequence[float] | np.ndarray,
                     label: str,
                     period: tuple = ("", ""),
                     source: str = "",
                     window: Optional[int] = None,
                     **fit_kwargs) -> CrisisCalibration:
    """Estimate a ``CrisisCalibration`` from a crisis-period return series.

    ``returns_pct`` should be the percent log returns *within* the crisis
    window (e.g. the ASPI or the stock over 2022).  ``window=None`` uses all
    of them; at least 100 observations are required by the estimator.
    """
    fit = fit_gjr_garch(returns_pct, window=window, **fit_kwargs)
    return CrisisCalibration(params=fit.params, label=label, period=tuple(period), source=source,
                             is_placeholder=False, log_likelihood=fit.log_likelihood, n_obs=fit.n_obs)


# --------------------------------------------------------------------------- #
# Scenario definitions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VolatilityMultiplier:
    """Scale the initial conditional volatility sigma_{T+1} by ``factor`` (> 0)."""
    factor: float

    def __post_init__(self):
        if self.factor <= 0:
            raise ValueError("factor must be > 0")

    @property
    def name(self) -> str:
        return f"volatility multiplier x{self.factor:g}"


@dataclass(frozen=True)
class ReturnShock:
    """Impose an initial return of ``shock_pct`` percent (normally negative)."""
    shock_pct: float

    def __post_init__(self):
        if self.shock_pct > 0:
            warnings.warn("ReturnShock with a positive shock: the scenario is meant for negative returns",
                          stacklevel=2)

    @property
    def name(self) -> str:
        return f"return shock {self.shock_pct:+g}%"


@dataclass(frozen=True)
class CrisisScenario:
    """Use crisis-calibrated parameters.  ``use_current_state`` keeps the
    current fit's (r_T, sigma2_T) instead of the crisis unconditional variance."""
    calibration: CrisisCalibration
    use_current_state: bool = False

    @property
    def name(self) -> str:
        return f"crisis scenario [{self.calibration.label}]"


Scenario = Union[VolatilityMultiplier, ReturnShock, CrisisScenario]


@dataclass(frozen=True)
class StressResult:
    scenario_name: str
    baseline: VaRResult
    stressed: VaRResult
    is_placeholder: bool = False
    notes: tuple = field(default_factory=tuple)

    def uplift(self, horizon: int, confidence: float) -> float:
        """stressed VaR / baseline VaR (percent units)."""
        b = self.baseline.get(horizon, confidence).var_pct
        s = self.stressed.get(horizon, confidence).var_pct
        return float("inf") if b == 0 else s / b

    def __str__(self) -> str:
        head = f"STRESS TEST: {self.scenario_name}"
        if self.is_placeholder:
            head += "\n  *** " + PLACEHOLDER_LABEL + " ***"
        rows = [head, f"{'h':>3} {'conf':>6} {'base VaR%':>10} {'stress VaR%':>12} {'x':>6} {'stress VaR LKR':>15}"]
        for (h, c), e in self.stressed.estimates.items():
            rows.append(f"{h:>3} {c:>6.2%} {self.baseline.get(h, c).var_pct:>10.4f} {e.var_pct:>12.4f} "
                        f"{self.uplift(h, c):>6.2f} {e.var_lkr:>15,.0f}")
        for n in self.notes:
            rows.append("  note: " + n)
        return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_stress_test(fit: GJRGARCHFit,
                    scenario: Scenario,
                    notional: float,
                    horizons: Sequence[int] = DEFAULT_HORIZONS,
                    confidence: Sequence[float] = DEFAULT_CONFIDENCE,
                    n_paths: int = DEFAULT_N_PATHS,
                    seed: Optional[int] = None,
                    baseline: Optional[VaRResult] = None) -> StressResult:
    """Apply one stress scenario to a fitted model and return baseline vs
    stressed VaR/ES.  The same seed is used for both so the difference is
    attributable to the scenario, not to sampling noise.

    ``baseline`` may be passed in to avoid recomputing it (e.g. in the
    dashboard where the baseline is already on screen).
    """
    common = dict(notional=notional, horizons=horizons, confidence=confidence, n_paths=n_paths, seed=seed)
    if baseline is None:
        baseline = estimate_var(fit, **common)

    notes: list[str] = []
    is_placeholder = False
    if isinstance(scenario, VolatilityMultiplier):
        stressed = estimate_var(fit, sigma2_multiplier=scenario.factor ** 2, **common)
        notes.append(f"sigma_(T+1) scaled from {fit.next_sigma:.4f}% to {fit.next_sigma * scenario.factor:.4f}%")
    elif isinstance(scenario, ReturnShock):
        stressed = estimate_var(fit, initial_return=scenario.shock_pct, **common)
        notes.append("shock applied as r_T (day 0); not included in the h-day return")
    elif isinstance(scenario, CrisisScenario):
        cal = scenario.calibration
        is_placeholder = cal.is_placeholder
        if scenario.use_current_state:
            r0, s2_0 = fit.last_return, fit.last_sigma2
            notes.append("crisis parameters started from the current fit's (r_T, sigma2_T)")
        else:
            r0, s2_0 = 0.0, cal.start_sigma2
            notes.append(f"crisis parameters started from crisis unconditional variance "
                         f"({np.sqrt(s2_0):.4f}% daily vol)")
        stressed = estimate_var(cal.params, initial_return=r0, initial_sigma2=s2_0, **common)
        if is_placeholder:
            notes.append(PLACEHOLDER_LABEL)
    else:
        raise TypeError(f"unknown scenario type {type(scenario).__name__}")

    return StressResult(scenario_name=scenario.name, baseline=baseline, stressed=stressed,
                        is_placeholder=is_placeholder, notes=tuple(notes))
