"""Run M6 (Risk Engine) on real CSE returns: rolling out-of-sample 1-day VaR + Kupiec (FR14), current
1-/5-day VaR at 95/99% (FR12), the 2022-crisis calibration and stored GJR-GARCH parameters for the
dashboard's live stress tests (FR13).

Returns are percent log returns of each security's corporate-action-adjusted close over its traded
days. The out-of-sample period is the walk-forward test period (calendar position >= 755), so VaR and
direction are evaluated over the same days. Parameters are re-estimated daily on the previous 750
returns (§4.6.5; cse_risk_engine D11 default).

Usage: python scripts/run_risk.py [--refit-every 1] [--paths 10000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cse_risk_engine import (  # noqa: E402
    ConvergenceError, calibrate_crisis, estimate_var, fit_gjr_garch, kupiec_pof_test, next_variance,
)
from models.universe import load_universe  # noqa: E402
from models import config as C  # noqa: E402

OUT = ROOT / "work" / "results"
CRISIS = ("2022-01-01", "2022-12-31")
NOTIONAL = 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-every", type=int, default=1)
    ap.add_argument("--paths", type=int, default=10_000)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    panel_all = pd.read_parquet(ROOT / "work" / "panel_all.parquet")
    panel, info = load_universe(panel_all)
    sids = sorted(panel["security_id"].unique())
    oos_start_pos = C.INITIAL_TRAIN_DAYS + C.PURGE_DAYS
    rows_bt, rows_k, rows_var, rows_par, rows_crisis, notes = [], [], [], [], [], []
    only = set(a.only) if a.only else None
    done = set()
    if (OUT / "kupiec_by_stock.parquet").exists() and not a.only:   # resume after an interruption
        k0 = pd.read_parquet(OUT / "kupiec_by_stock.parquet")
        done = set(k0["security_id"]) - {"POOLED"}
        keep = lambda f: [r for r in pd.read_parquet(OUT / f).to_dict("records") if r["security_id"] in done] if (OUT / f).exists() else []
        rows_bt, rows_k, rows_var, rows_par = keep("var_backtest.parquet"), keep("kupiec_by_stock.parquet"), keep("var_daily.parquet"), keep("garch_params_daily.parquet")
        rows_crisis = keep("crisis_params.parquet")
        print(f"resuming: {len(done)} securities already done", flush=True)
    print(f"risk: {info}; {len(sids)} securities", flush=True)
    for sid in sids:
        if (only and sid not in only) or sid in done:
            continue
        t0 = time.time()
        g = panel_all[panel_all["security_id"] == sid].sort_values("cal_pos")
        g = g[g["lr"].notna()]
        r = g["lr"].values * 100
        dates, pos = g["date"].values, g["cal_pos"].values
        start = max(int(np.searchsorted(pos, oos_start_pos)), 750)
        if len(r) < 850 or start >= len(r):
            notes.append(f"{sid}: too few observations ({len(r)})")
            continue
        fit, failed, nonstat, breaches = None, 0, 0, {0.95: [], 0.99: []}
        for k, t in enumerate(range(start, len(r))):
            if fit is None or k % a.refit_every == 0:
                try:
                    try:
                        fit = fit_gjr_garch(r, window=750, end=t)
                    except ConvergenceError:
                        # author decision 2026-10-13: a fit at the persistence bound (IGARCH-like) is
                        # accepted and flagged instead of carrying stale parameters
                        fit = fit_gjr_garch(r, window=750, end=t, strict=False)
                        nonstat += 1
                    state = (fit.params, fit.last_return, fit.last_sigma2)
                except ConvergenceError:
                    failed += 1
                    if fit is None:
                        continue
                    # carry the previous parameters, update the variance one step through eq. 4.5
                    prm, r_prev, s2_prev = state
                    state = (prm, r[t - 1], next_variance(prm, state[1], state[2]))
            else:
                prm, r_prev, s2_prev = state
                state = (prm, r[t - 1], next_variance(prm, r_prev, s2_prev))
            prm, r_prev, s2_prev = state
            vr = estimate_var(prm, NOTIONAL, horizons=(1, 5), confidence=(0.95, 0.99), n_paths=a.paths,
                              seed=C.SEED + t, initial_return=r_prev, initial_sigma2=s2_prev)
            # the VaR for day t is known after the close of day t-1 (as_of = previous traded day)
            as_of = dates[t - 1]
            for (h, c), e in vr.estimates.items():
                rows_var.append({"security_id": sid, "as_of": as_of, "horizon_days": h, "confidence": c,
                                 "var_pct": e.var_pct, "es_pct": e.es_pct})
            for c in (0.95, 0.99):
                v1 = vr.estimates[(1, c)].var_pct
                b = bool(r[t] < -v1)
                breaches[c].append(b)
                rows_bt.append({"security_id": sid, "date": dates[t], "as_of": as_of, "confidence": c,
                                "realised_pct": r[t], "var_pct": v1, "breach": b})
            d = prm.as_dict()
            d.update({"security_id": sid, "as_of": as_of, "last_return": r_prev, "last_sigma2": s2_prev,
                      "persistence": prm.persistence, "nonstationary": bool(prm.persistence >= 1)})
            rows_par.append(d)
        for c in (0.95, 0.99):
            k = kupiec_pof_test(np.array(breaches[c]), confidence=c).as_dict()
            k.update({"security_id": sid, "confidence": c, "failed_fits": failed, "nonstationary_fits": nonstat})
            rows_k.append(k)
        # state after the final day (as_of = last trading day) for "today's" VaR in the dashboard
        try:
            try:
                fit = fit_gjr_garch(r, window=750)
            except ConvergenceError:
                fit = fit_gjr_garch(r, window=750, strict=False)
            vr = estimate_var(fit, NOTIONAL, horizons=(1, 5), confidence=(0.95, 0.99), n_paths=a.paths, seed=C.SEED)
            for (h, c), e in vr.estimates.items():
                rows_var.append({"security_id": sid, "as_of": dates[-1], "horizon_days": h, "confidence": c,
                                 "var_pct": e.var_pct, "es_pct": e.es_pct})
            d = fit.params.as_dict()
            d.update({"security_id": sid, "as_of": dates[-1], "last_return": fit.last_return,
                      "last_sigma2": fit.last_sigma2, "persistence": fit.params.persistence,
                      "std_errors": json.dumps(fit.std_errors)})
            rows_par.append(d)
        except ConvergenceError as e:
            notes.append(f"{sid}: final fit failed: {e}")
        cm = (dates >= np.datetime64(CRISIS[0])) & (dates <= np.datetime64(CRISIS[1]))
        try:
            cc = calibrate_crisis(r[cm], label=f"{sid} 2022 crisis", period=CRISIS, source="CSE Data Library", strict=False)
            d = cc.params.as_dict()
            d.update({"security_id": sid, "n_obs": cc.n_obs, "log_likelihood": cc.log_likelihood,
                      "unconditional_sigma2": cc.params.unconditional_variance if cc.params.is_stationary else np.nan})
            rows_crisis.append(d)
        except Exception as e:  # noqa: BLE001
            notes.append(f"{sid}: crisis calibration failed: {e}")
        if failed:
            notes.append(f"{sid}: {failed} daily fits failed (previous parameters carried)")
        print(f"{sid}: {len(breaches[0.95])} OOS days, breaches 95%={sum(breaches[0.95])} 99%={sum(breaches[0.99])}, "
              f"non-stationary fits accepted={nonstat}, failed fits={failed} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows_bt).to_parquet(OUT / "var_backtest.parquet")
        pd.DataFrame(rows_k).to_parquet(OUT / "kupiec_by_stock.parquet")
        pd.DataFrame(rows_var).to_parquet(OUT / "var_daily.parquet")
        pd.DataFrame(rows_par).to_parquet(OUT / "garch_params_daily.parquet")
        pd.DataFrame(rows_crisis).to_parquet(OUT / "crisis_params.parquet")
    bt = pd.DataFrame(rows_bt)
    pooled = []
    for c in (0.95, 0.99):
        b = bt[bt["confidence"] == c]["breach"].values
        k = kupiec_pof_test(b, confidence=c).as_dict()
        k.update({"security_id": "POOLED", "confidence": c})
        pooled.append(k)
    pd.DataFrame(rows_k + pooled).to_parquet(OUT / "kupiec_by_stock.parquet")
    pd.DataFrame(rows_var).to_parquet(OUT / "var_daily.parquet")
    pd.DataFrame(rows_par).to_parquet(OUT / "garch_params_daily.parquet")
    pd.DataFrame(rows_crisis).to_parquet(OUT / "crisis_params.parquet")
    json.dump({"universe": info, "notes": notes, "refit_every": a.refit_every, "paths": a.paths},
              open(OUT / "risk_run.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
