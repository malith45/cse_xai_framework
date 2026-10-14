# cse_risk_engine — Module M6 (Risk Engine)

Standalone, importable Python package implementing Chapter 4, §4.6.5 of the
dissertation *Explainable Machine Learning Framework for Stock Movement and Risk
Prediction on the Colombo Stock Exchange*: GJR-GARCH(1,1)-t volatility,
Monte Carlo Value-at-Risk and Expected Shortfall, three stress scenarios and the
Kupiec (1995) proportion-of-failures backtest.

* Pure-function core. Depends only on `numpy`, `scipy`, `pandas`, `arch`.
* No Streamlit, no database, no other framework module. The dashboard (M8) and
  the offline pipeline (M7) import it.
* Tested against synthetic returns with known properties (66 tests, ~6 s).

```
cse_risk_engine/
├── cse_risk_engine/
│   ├── __init__.py      public API
│   ├── volatility.py    GJR-GARCH(1,1)-t fit (arch), recursion, filtering, helpers
│   ├── simulation.py    vectorised Monte Carlo paths
│   ├── var.py           VaR / ES, percent → LKR
│   ├── stress.py        volatility multiplier, return shock, crisis calibration
│   ├── kupiec.py        proportion-of-failures test
│   └── backtest.py      rolling out-of-sample 1-day VaR → Kupiec
├── tests/               pytest suite (see "Tests")
├── example.py           end-to-end demo on synthetic data
├── requirements.txt     pinned versions
└── pyproject.toml
```

## Install and test

```bash
pip install -r requirements.txt
pip install -e .          # makes `import cse_risk_engine` work anywhere
pytest                    # runs tests/
python example.py         # prints a full fit → VaR → stress → backtest run
```

---

## UNIT CONVENTION — read this first

Every public function uses **the same unit**. This is the single most likely
source of a silent wrong answer, so it is fixed once and never varied:

| Quantity | Unit | Example |
|---|---|---|
| Returns (input and simulated) | **daily log returns in percent**: `100 · ln(P_t / P_{t-1})` | `1.25` = +1.25 % |
| `omega`, `sigma2`, `last_sigma2`, `next_sigma2` | percent² | `0.45` ⇒ σ = 0.67 % daily |
| `alpha`, `gamma`, `beta` | dimensionless | |
| `nu` | Student-t degrees of freedom (> 2) | |
| `var_pct`, `es_pct` | **positive** percent loss in log-return units | `1.79` = 1.79 % |
| `var_lkr`, `es_lkr` | LKR loss on the notional, **exact**: `notional · (1 − e^(−var_pct/100))` | |
| `notional` | LKR | |
| `horizon` | trading days | 1 or 5 |
| `confidence` | decimal | 0.95, 0.99 |
| `expected_rate` (Kupiec `p`) | decimal, `= 1 − confidence` | 0.05 |

Build the input series with `log_returns_pct(prices)`. If you pass decimals by
mistake (std ≈ 0.007 instead of ≈ 0.7) `fit_gjr_garch` emits a `UserWarning`
that says so.

Why exact LKR conversion: Chapter 4 data-preparation step (6) defines returns
as log returns and VaR as a percentile of the h-day (log) return distribution.
The loss on a position whose log return is −v % is exactly `1 − e^(−v/100)`
of the notional; `notional · v/100` is a first-order approximation that
overstates the loss (by ≈ 0.5 % of notional at v = 10 %). The exact form is
the only one consistent with the text.

---

## Modelling decisions (traceable to the chapters)

| # | Decision | Where it comes from |
|---|---|---|
| D1 | **Zero mean equation.** | Chapter 4 specifies no mean equation and eq. 4.5 conditions the indicator on the *prior return* while writing the recursion in ε². Both hold together only if ε_t = r_t, i.e. zero mean. `arch_model(mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)`. If a constant mean is ever adopted, eq. 4.5 must be reworded to "prior residual". |
| D2 | Innovations: standardised Student-t, `z = t(ν)·√((ν−2)/ν)`, unit variance — identical to `arch`'s parametrisation, so simulation and estimation use the same distribution. | §4.6.5, Bollerslev (1987) |
| D3 | Window: last 750 trading days; `end` parameter walks the window forward. | §4.6.5 |
| D4 | Non-convergence: up to 3 optimiser attempts with different starting values; then `ConvergenceError` (with `arch`'s convergence flag and the offending parameters in the message). A fit with persistence ≥ 1 is also rejected unless `strict=False`. Nothing is returned silently. | spec §1 |
| D5 | Day-1 simulated variance is the deterministic `σ²_{T+1}` from eq. 4.5 applied to `(r_T, σ²_T)` — verified equal to `arch`'s own 1-step forecast to 1e-10. | §4.6.5 |
| D6 | VaR quantile: `numpy.percentile` (linear interpolation) of the h-day cumulative log return; VaR = −percentile. ES = mean of paths at or below −VaR. | §4.6.5, spec §3 |
| D7 | Volatility multiplier `k` scales **σ_{T+1}** (day-1 volatility), i.e. variance by `k²`; the recursion then mean-reverts, so 1-day VaR scales exactly by `k` and 5-day VaR by slightly less. | spec §4(a) |
| D8 | Return shock `s` is imposed as `r_T` (the return just realised, day 0). It raises `σ²_{T+1}` via `(α+γ)·s²` and is **not** counted inside the h-day return: the output is the conditional VaR *after* the shock. | spec §4(b) |
| D9 | Crisis scenario replaces the parameters with a `CrisisCalibration`; by default starts from the crisis model's unconditional variance ("the market behaves like the crisis from today"); `use_current_state=True` keeps `(r_T, σ²_T)`. The only built-in object is `CrisisCalibration.placeholder(fit)`, which reuses the current fit's parameters and is flagged `is_placeholder=True` in every result and printout. **No crisis numbers are hard-coded.** | spec §4(c) |
| D10 | Kupiec edge cases: `x = 0 ⇒ LR = −2T·ln(1−p)`, `x = T ⇒ LR = −2T·ln p` (exact limits, 0⁰ := 1). Verdict "pass" = not rejected at 5 % (`p-value ≥ 0.05`, equivalently `LR < 3.841`). | eq. 4.6, spec §5 |
| D11 | Rolling backtest refits daily by default (`refit_every=1`, as the text implies); `refit_every=k` is available for speed and between refits the variance is still updated daily through eq. 4.5. Per-day seed = `seed + t`. | §4.6.5 |

---

## API reference

### `volatility.py`

**`log_returns_pct(prices) -> ndarray`** — `100·ln(P_t/P_{t−1})`, length `n−1`. Raises on non-positive prices.

**`GJRGARCHParams(omega, alpha, gamma, beta, nu)`** — frozen dataclass; validates `omega>0`, `alpha,beta≥0`, `alpha+gamma≥0`, `nu>2`. Properties: `persistence = α + γ/2 + β`, `is_stationary`, `unconditional_variance = ω/(1−persistence)` (percent²), `as_array()`, `as_dict()`.

**`fit_gjr_garch(returns_pct, window=750, *, end=None, max_attempts=3, maxiter=1000, strict=True) -> GJRGARCHFit`**

| | |
|---|---|
| `returns_pct` | 1-D percent log returns |
| `window` | observations used (`returns_pct[end-window:end]`); `None` = whole series |
| `end` | exclusive end index (rolling use); default `len(returns_pct)` |
| `strict` | reject persistence ≥ 1 as a failure |
| Raises | `ValueError` (bad input: < 100 obs, NaN, zero variance), `ConvergenceError` |

`GJRGARCHFit` fields: `params: GJRGARCHParams`, `std_errors: dict` (robust s.e. keyed `omega…nu`), `log_likelihood`, `aic`, `bic`, `converged` (always `True` on a returned object), `n_obs`, `window: (start, end)`, `last_return` (`r_T`, %), `last_sigma2` (`σ²_T`, %²), `warnings: tuple[str]`. Properties `next_sigma2`, `next_sigma` (`σ_{T+1}`), method `summary()`.

**`next_variance(params, prev_return, prev_sigma2)`** — one step of eq. 4.5; vectorised over arrays.

**`filter_variance(params, returns_pct, sigma2_0=None) -> ndarray`** — run the recursion through observed returns (no estimation). Reproduces `arch`'s conditional variance to 1e-8 given the same `sigma2_0`.

### `simulation.py`

**`simulate_paths(model, horizon=1, n_paths=10_000, seed=None, *, initial_return=None, initial_sigma2=None, sigma2_multiplier=1.0) -> SimulationResult`**

`model` is a `GJRGARCHFit` (state taken from the fit) or bare `GJRGARCHParams` (then `initial_return`, `initial_sigma2` are required). `SimulationResult`: `returns (n_paths, horizon)` %, `cumulative (n_paths,)` % (row sums), `sigma2 (n_paths, horizon)` %², `initial_sigma2` (day-1 variance actually used), `initial_return`, `params`, settings. 10,000 × 5 paths ≈ 5 ms.

**`simulate_gjr_garch_series(params, n_obs, seed=None, burn_in=500)`** — one long synthetic series from known parameters (for tests / demos).

### `var.py`

**`value_at_risk(cumulative_returns_pct, confidence) -> float`** — positive % loss = −percentile at `100·(1−confidence)`.
**`expected_shortfall(cumulative_returns_pct, confidence) -> float`** — mean loss over paths with return ≤ −VaR.
**`log_loss_to_lkr(var_pct, notional) -> float`** — `notional·(1−e^(−var_pct/100))`.

**`estimate_var(model, notional, horizons=(1,5), confidence=(0.95,0.99), n_paths=10_000, seed=None, **sim_kwargs) -> VaRResult`** — one simulation per horizon (same seed reused), all confidence levels from each. `VaRResult.get(h, c) -> VaREstimate(horizon, confidence, var_pct, var_lkr, es_pct, es_lkr, notional, n_paths)`; `.table()` gives a list of dicts for the dashboard; `str()` prints a table.

### `stress.py`

Scenarios (frozen dataclasses): **`VolatilityMultiplier(factor)`**, **`ReturnShock(shock_pct)`** (warns if positive), **`CrisisScenario(calibration, use_current_state=False)`**.

**`CrisisCalibration(params, label, period=("",""), source="", initial_sigma2=None, is_placeholder=False, log_likelihood=None, n_obs=None)`** — the swappable interface. `CrisisCalibration.placeholder(fit)` builds the flagged stand-in. **`calibrate_crisis(returns_pct, label, period, source, window=None) -> CrisisCalibration`** fits the model on the crisis-window returns (≥ 100 observations) — this is the call to make once the 2022 CSE window is available.

**`run_stress_test(fit, scenario, notional, horizons, confidence, n_paths, seed, baseline=None) -> StressResult`** — baseline and stressed `VaRResult` with the *same seed* (differences are attributable to the scenario), `uplift(h, c)` = stressed/baseline VaR, `is_placeholder`, `notes`. Pass `baseline` to skip recomputing it.

### `kupiec.py`

**`kupiec_pof_test(breaches=None, *, n_obs=None, n_breaches=None, confidence=0.95, significance=0.05) -> KupiecResult`** — supply a boolean/0-1 breach sequence *or* the counts. Result fields: `n_obs (T)`, `n_breaches (x)`, `observed_rate`, `expected_rate (p)`, `lr_statistic`, `p_value`, `significance`, `critical_value`, `passed`, property `verdict`, `as_dict()`.
**`kupiec_lr_statistic(n_obs, n_breaches, p) -> float`** — the bare statistic with edge cases handled.

### `backtest.py`

**`rolling_var_backtest(returns_pct, window=750, confidence=(0.95,0.99), *, horizon=1, refit_every=1, n_paths=10_000, seed=None, start=None, end=None, significance=0.05, on_fit_failure="raise") -> BacktestResult`**

For each out-of-sample day `t`: fit on `[t−window, t)`, simulate from `(r_{t−1}, σ²_{t−1})`, breach if `r_t < −VaR_t`. `BacktestResult`: `index`, `realised`, `var {conf: ndarray}`, `breaches {conf: bool ndarray}`, `kupiec {conf: KupiecResult}`, `refit_index`, `failed_fits`. `on_fit_failure="carry"` keeps the previous parameters, records the day and warns; there is no silent option. Cost ≈ 0.15–0.5 s per refit on a laptop CPU, so daily refitting over 750 out-of-sample days is a few minutes — run it in the offline pipeline, not the dashboard.

---

## Tests (`pytest`, 66 tests)

| File | What is proven |
|---|---|
| `test_volatility.py` | Parameter recovery from a known GJR-GARCH-t process (5,000 obs, every parameter within 3 s.e. *and* an absolute tolerance; repeated on 3 independent 8,000-obs series); `next_variance` equals `arch`'s 1-step forecast to 1e-10; `filter_variance` reproduces `arch`'s conditional variance; 750-window and `end` indexing; unit warning on decimal input; `ConvergenceError` on flag ≠ 0, on optimiser exceptions and on non-stationary fits (monkeypatched); parameter validation. |
| `test_simulation.py` | Same seed ⇒ identical output; different seed ⇒ different; 10,000-path baseline + two stress tests < 10 s (a single 10k×5 simulation < 1 s); **asymmetry**: negative shock raises next variance by exactly `γ·r²` more than a positive one, propagates through 5-day simulated variance and the loss tail, and vanishes when `γ = 0`; standardised-t has unit variance; day-1 variance = `next_sigma2`; day-2 variance obeys eq. 4.5 path by path; multiplier scales variance exactly; generator's sample variance ≈ `ω/(1−persistence)`. |
| `test_var.py` | MC VaR/ES within 2–3 % of analytic Normal VaR/ES for h = 1 and 5 (`ν = 10⁷`, 400k paths); MC VaR matches analytic Student-t(5) VaR (checks the `(ν−2)/ν` scaling); hand-checked percentile/ES arithmetic; ES ≥ VaR, 99 % ≥ 95 %, 5-day > 1-day; exact LKR conversion (`10 % → 95,162.58 on 1 m`); percent is notional-free, LKR scales linearly. |
| `test_kupiec.py` | **Worked example — Jorion (2007) ch. 6, JPMorgan 1998: T = 252, x = 20, p = 0.05 ⇒ LR = 3.91, reject** (matches the textbook to 2 d.p.); **Kupiec (1995) Table 3 non-rejection regions** (p = 0.05: T = 255 ⇒ 6 < N < 21, T = 510 ⇒ 16 < N < 36, T = 1000 ⇒ 37 < N < 65; p = 0.01, T = 255 ⇒ N < 7) reproduced exactly at both boundaries; x/T = p ⇒ LR = 0; hand-computed T = 100, x = 10 ⇒ LR = 4.131; **x = 0 and x = T edge cases** equal the closed-form limits and are finite; LR is non-negative and V-shaped around T·p; breach-sequence and count interfaces agree; p-value/verdict consistent with χ²(1). |
| `test_stress.py` | Multiplier scales 1-day VaR *exactly* by the factor, 5-day by 1.5–2×; return shock raises VaR monotonically and the day-1 ratio equals `√(eq. 4.5 / next_sigma2)` exactly; negative shock > positive shock; placeholder flagged everywhere; `calibrate_crisis` on a synthetic high-volatility "crisis" yields a non-placeholder calibration with > 1.5× VaR; explicit `initial_sigma2` honoured. |
| `test_backtest.py` | Shapes, breach/VaR consistency, Kupiec reproduced from the breach vector; correctly-specified model passes Kupiec over 1,750 out-of-sample days; **no look-ahead** (corrupting future returns leaves earlier VaRs bit-identical, and the first affected day differs); seed reproducibility; a fit failure raises, or with `"carry"` warns and is recorded — never silent. |

---

## Known limitations / to revisit with real data

1. `arch` bounds persistence at 1.0; CSE series with near-IGARCH behaviour may hit that bound and be rejected in `strict` mode. If that happens on real data, decide explicitly whether to (a) accept with `strict=False` and report the warning, or (b) fall back to a longer window — do not just switch `strict` off globally.
2. GARCH parameters are individually noisy on 750 observations (β and γ trade off); persistence and the resulting VaR are much better determined. Report standard errors from `std_errors` in Chapter 5.
3. `CrisisCalibration.placeholder` must be replaced by `calibrate_crisis(...)` on the actual 2022 window before any crisis-scenario figure appears in the dissertation.
4. The Kupiec test is an unconditional-coverage test only; Christoffersen's independence test is not implemented (not in the specification).

## References

Bollerslev, T. (1987). *Rev. Econ. Stat.*, 69(3). · Glosten, Jagannathan & Runkle (1993). *J. Finance*, 48(5). · Jorion, P. (2007). *Value at Risk* (3rd ed.), ch. 6. · Kupiec, P. (1995). *J. Derivatives*, 3(2). · Sheppard, K. *arch* library documentation.
