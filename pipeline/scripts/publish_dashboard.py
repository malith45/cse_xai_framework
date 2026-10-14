"""Build the dashboard's read-only results tables (M8 data layer) from the offline outputs.

Writes dist/db/<table>.parquet and dist/db/cse_dashboard.sqlite (for local runs and screenshots).
scripts/load_to_postgres.py pushes the same tables to Neon. Only derived results, adjusted prices of
the modelling universe and index levels are published; no raw licensed file and no article body
text ever leaves the offline environment (Chapter 4 §4.7 step 14).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models.explain import LABELS  # noqa: E402
from models.universe import load_universe  # noqa: E402

WORK, RES = ROOT / "work", ROOT / "work" / "results"
OUT = ROOT / "dist" / "db"
HZ = {"t1": "t+1", "t5": "t+5"}
MODEL_VERSION = "xgb-wf-v1"


def ticker(sid: str) -> str:
    return sid.split("-")[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(WORK / "prices_adj.parquet")
    ix = pd.read_parquet(WORK / "indices.parquet")
    panel_all = pd.read_parquet(WORK / "panel_all.parquet")
    panel, uni = load_universe(panel_all)
    sids = sorted(panel["security_id"].unique())
    cal = pd.DatetimeIndex(sorted(ix["date"].unique()))
    tables: dict[str, pd.DataFrame] = {}

    names = prices.groupby("security_id")["short_name"].last()
    last_day = panel["date"].max()
    current = set(panel.loc[panel["date"] == last_day, "security_id"])
    comp = pd.DataFrame({"ticker": [ticker(s) for s in sids], "security_id": sids,
                         "name": [str(names.get(s, s)).title() for s in sids],
                         "sector": ["S&P SL20 now" if s in current else "Former S&P SL20" for s in sids]})
    comp["sort_order"] = [(0 if s in current else 1, ticker(s)) for s in sids]
    comp = comp.sort_values("sort_order").reset_index(drop=True)
    comp["sort_order"] = range(len(comp))
    tables["company"] = comp
    tables["index_level"] = ix[["date", "aspi", "sl20"]].copy()
    p = prices[prices["security_id"].isin(sids)]
    tables["daily_price"] = pd.DataFrame({"ticker": p["security_id"].map(ticker), "date": p["date"],
                                          "open": p["adj_open"].round(4), "high": p["adj_high"].round(4),
                                          "low": p["adj_low"].round(4), "close": p["adj_close"].round(4),
                                          "volume": p["adj_shares"].round(0)})

    # market breadth over every ordinary voting share (an aggregate count, not raw data)
    allp = prices.sort_values(["security_id", "date"]).copy()
    allp["chg"] = allp.groupby("security_id")["adj_close"].pct_change()
    br = allp.dropna(subset=["chg"]).groupby("date")["chg"].agg(
        advances=lambda s: int((s > 1e-9).sum()), declines=lambda s: int((s < -1e-9).sum()),
        unchanged=lambda s: int((s.abs() <= 1e-9).sum())).reset_index()
    tables["market_breadth"] = br

    # predictions, SHAP, narratives (XGBoost = published model)
    xp = pd.read_parquet(RES / "xgb_predictions.parquet")
    pos = pd.Series(np.arange(len(cal)), index=cal)

    def target(d, h):
        i = pos.get(pd.Timestamp(d))
        if i is not None and i + h < len(cal):
            return cal[i + h]
        return pd.bdate_range(d, periods=h + 1)[-1]

    xp["target_date"] = [target(d, 1 if hz == "t1" else 5) for d, hz in zip(xp["date"], xp["horizon"])]
    tables["prediction"] = pd.DataFrame({
        "ticker": xp["security_id"].map(ticker), "as_of": xp["date"], "horizon": xp["horizon"].map(HZ),
        "model": "XGBoost", "p_up": xp["p_up"].round(4), "threshold": xp["threshold"].round(4),
        "direction": np.where(xp["pred"] == 1, "up", "down"), "target_date": xp["target_date"],
        "model_version": MODEL_VERSION, "block": xp["block"], "narrative": xp["narrative"]})
    sh = pd.read_parquet(RES / "xgb_shap_top10.parquet")
    tables["shap_contribution"] = pd.DataFrame({
        "ticker": sh["security_id"].map(ticker), "as_of": sh["date"], "horizon": sh["horizon"].map(HZ),
        "rank": sh["rank"], "feature": sh["feature"].map(lambda f: LABELS.get(f, f)),
        "value": sh["feature_value"].map(lambda v: v if isinstance(v, str) else (None if pd.isna(v) else round(float(v), 4))).astype(str),
        "shap": sh["shap_value"].round(5), "base_value": sh["base_value"].round(5)})
    g = pd.read_parquet(RES / "xgb_shap_global.parquet")
    g = g.groupby(["horizon", "feature"])["mean_abs_shap"].mean().reset_index()
    tables["global_importance"] = pd.DataFrame({"horizon": g["horizon"].map(HZ),
                                                "feature": g["feature"].map(lambda f: LABELS.get(f, f)),
                                                "mean_abs_shap": g["mean_abs_shap"].round(5)})

    # per-stock accuracy by block (out-of-sample)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    rows = []
    ok = xp[xp["label"].notna()]
    for (sid, hz, b), d in ok.groupby(["security_id", "horizon", "block"]):
        y, pr = d["label"].astype(int), d["pred"].astype(int)
        rows.append({"ticker": ticker(sid), "horizon": HZ[hz], "block": int(b) + 1, "test_start": d["date"].min(),
                     "test_end": d["date"].max(), "accuracy": accuracy_score(y, pr),
                     "f1": f1_score(y, pr, average="macro", zero_division=0),
                     "precision": precision_score(y, pr, zero_division=0), "recall": recall_score(y, pr, zero_division=0),
                     "n": len(d)})
    tables["block_accuracy"] = pd.DataFrame(rows)

    # risk
    if (RES / "var_daily.parquet").exists():
        v = pd.read_parquet(RES / "var_daily.parquet")
        tables["var_estimate"] = pd.DataFrame({"ticker": v["security_id"].map(ticker), "as_of": v["as_of"],
                                               "horizon_days": v["horizon_days"], "confidence": v["confidence"],
                                               "var_pct": v["var_pct"].round(4)})
        gp = pd.read_parquet(RES / "garch_params_daily.parquet")
        crisis = pd.read_parquet(RES / "crisis_params.parquet").set_index("security_id") if (RES / "crisis_params.parquet").exists() else pd.DataFrame()
        r22 = panel_all[(panel_all["date"].dt.year == 2022)].groupby("security_id")["lr"].quantile(0.01)
        rp = pd.DataFrame({"ticker": gp["security_id"].map(ticker), "as_of": gp["as_of"], "mu": 0.0,
                           "omega": gp["omega"] / 1e4, "alpha": gp["alpha"], "gamma": gp["gamma"], "beta": gp["beta"],
                           "nu": gp["nu"], "sigma_last": np.sqrt(gp["last_sigma2"]) / 100, "ret_last": gp["last_return"] / 100})
        if len(crisis):
            cu = crisis["unconditional_sigma2"].reindex(gp["security_id"]).values
            mult = np.sqrt(cu / gp["last_sigma2"].values)
            rp["crisis_vol_multiplier"] = np.clip(np.nan_to_num(mult, nan=2.0), 1.0, 4.0).round(2)
        else:
            rp["crisis_vol_multiplier"] = 2.0
        rp["crisis_shock"] = gp["security_id"].map(r22).fillna(-0.05).round(4).values
        tables["risk_parameters"] = rp
        k = pd.read_parquet(RES / "kupiec_by_stock.parquet")
        k = k[k["security_id"] != "POOLED"]
        tables["kupiec_result"] = pd.DataFrame({
            "ticker": k["security_id"].map(ticker), "confidence": k["confidence"], "observations": k["T"],
            "breaches": k["x"], "expected_breaches": (k["T"] * k["expected_rate"]).round(1),
            "lr_pof": k["LR_POF"].round(3), "p_value": k["p_value"].round(4),
            "verdict": np.where(k["passed"], "Accept", "Reject")})
        # market risk summary: median 1-day 95% VaR across the universe + ASPI realised volatility regime
        v95 = v[(v["horizon_days"] == 1) & (v["confidence"] == 0.95)].groupby("as_of")["var_pct"].median()
        a = np.log(ix.set_index("date")["aspi"]).diff()
        rv = a.rolling(20).std() * 100
        q1, q2 = rv.expanding(250).quantile(1 / 3), rv.expanding(250).quantile(2 / 3)
        regime = np.where(rv > q2, "Elevated", np.where(rv > q1, "Moderate", "Calm"))
        mr = pd.DataFrame({"date": rv.index, "realised_vol_20d": rv.round(3).values, "regime": regime})
        mr["var_1d_95"] = mr["date"].map(v95).round(3)
        tables["market_risk"] = mr.dropna(subset=["realised_vol_20d"])

    # sentiment (when M3 outputs exist)
    sp = RES / "articles_daily.parquet"
    if sp.exists():
        for name in ("sentiment_daily", "article"):
            f = RES / f"{name}.parquet"
            if f.exists():
                tables[name] = pd.read_parquet(f)

    as_of = xp["date"].max()
    tables["data_status"] = pd.DataFrame([{"as_of": as_of, "source": "CSE Data Library 2020-2025 (adjusted)",
                                           "model_version": MODEL_VERSION, "universe": uni}])

    con = sqlite3.connect(OUT / "cse_dashboard.sqlite")
    for name, df in tables.items():
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = pd.to_datetime(df[c]).dt.date
        df.to_parquet(OUT / f"{name}.parquet")
        df2 = df.copy()
        for c in df2.columns:
            if df2[c].map(lambda x: hasattr(x, "isoformat")).any():
                df2[c] = df2[c].map(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)
        df2.to_sql(name, con, if_exists="replace", index=False)
        print(f"{name}: {len(df)} rows")
    con.close()
    json.dump({"universe": uni, "as_of": str(as_of), "tables": list(tables)}, open(OUT / "manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
