"""M7 Evaluation: classification metrics per model, horizon and block (FR10), baseline comparison,
significance tests, and the Kupiec summary - written to work/results/evaluation.xlsx for Chapter 5.

Metrics on out-of-sample test blocks only. Rows with an undefined label (security did not trade on
the target day) are excluded and counted. Tests:
* exact binomial test of accuracy against 50% (coin toss);
* McNemar's exact test of each ML model against each baseline on the rows both predicted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "work" / "results"


def metrics(y, p, prob=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    out = {"n": len(y), "accuracy": accuracy_score(y, p), "precision_up": precision_score(y, p, zero_division=0),
           "recall_up": recall_score(y, p, zero_division=0), "f1_up": f1_score(y, p, zero_division=0),
           "f1_macro": f1_score(y, p, average="macro", zero_division=0),
           "balanced_accuracy": balanced_accuracy_score(y, p), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
           "share_up_actual": float(np.mean(y)), "share_up_predicted": float(np.mean(p))}
    out["binom_p_vs_50pct"] = binomtest(int(round(out["accuracy"] * len(y))), len(y), 0.5, alternative="greater").pvalue
    if prob is not None and len(np.unique(y)) == 2:
        out["auc"] = roc_auc_score(y, prob)
    return out


def mcnemar(y, a, b) -> tuple[int, int, float]:
    ca, cb = (a == y), (b == y)
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def main():
    frames = []
    for f in ("xgb_predictions.parquet", "rnn_predictions.parquet"):
        if (RES / f).exists():
            frames.append(pd.read_parquet(RES / f))
    ml = pd.concat(frames, ignore_index=True)
    base = pd.read_parquet(RES / "baseline_predictions.parquet") if (RES / "baseline_predictions.parquet").exists() else pd.DataFrame()
    labels = ml[["security_id", "date", "horizon", "label"]].drop_duplicates(["security_id", "date", "horizon"])
    long = [ml[["security_id", "date", "horizon", "block", "model", "pred", "p_up", "label"]]]
    if len(base):
        for hz in ("t1", "t5"):
            b = base[["security_id", "date", "block", "model", hz]].rename(columns={hz: "pred"}).assign(horizon=hz, p_up=np.nan)
            b = b.merge(labels, on=["security_id", "date", "horizon"], how="inner")
            long.append(b)
    # majority-class rule ("never up"): the accuracy any model must beat when up-days are a minority
    maj = ml[ml["model"] == ml["model"].iloc[0]][["security_id", "date", "horizon", "block", "label"]].copy()
    maj["model"], maj["pred"], maj["p_up"] = "always_not_up", 0, np.nan
    long.append(maj)
    allp = pd.concat(long, ignore_index=True)
    excluded = allp.groupby(["model", "horizon"])["label"].apply(lambda s: int(s.isna().sum())).rename("excluded_no_label")
    allp = allp[allp["label"].notna() & allp["pred"].notna()].copy()
    allp["label"], allp["pred"] = allp["label"].astype(int), allp["pred"].astype(int)

    overall = []
    for (m, hz), g in allp.groupby(["model", "horizon"]):
        r = metrics(g["label"].values, g["pred"].values, g["p_up"].values if g["p_up"].notna().all() else None)
        r.update({"model": m, "horizon": hz, "blocks": g["block"].nunique()})
        overall.append(r)
    overall = pd.DataFrame(overall).merge(excluded.reset_index(), on=["model", "horizon"], how="left")
    by_block = []
    for (m, hz, b), g in allp.groupby(["model", "horizon", "block"]):
        r = metrics(g["label"].values, g["pred"].values)
        r.update({"model": m, "horizon": hz, "block": b, "from": g["date"].min(), "to": g["date"].max()})
        by_block.append(r)
    by_block = pd.DataFrame(by_block)
    by_stock = []
    for (m, hz, s), g in allp[allp["model"].isin(["xgboost", "lstm", "gru"])].groupby(["model", "horizon", "security_id"]):
        r = metrics(g["label"].values, g["pred"].values)
        r.update({"model": m, "horizon": hz, "security_id": s})
        by_stock.append(r)
    by_stock = pd.DataFrame(by_stock)

    tests = []
    for hz in ("t1", "t5"):
        h = allp[allp["horizon"] == hz]
        piv = h.pivot_table(index=["security_id", "date"], columns="model", values="pred")
        y = h.drop_duplicates(["security_id", "date"]).set_index(["security_id", "date"])["label"]
        for mlm in [c for c in piv.columns if c in ("xgboost", "lstm", "gru")]:
            for bm in [c for c in piv.columns if c in ("naive", "arima", "garch")]:
                both = piv[[mlm, bm]].dropna()
                yy = y.reindex(both.index).values
                n01, n10, p = mcnemar(yy, both[mlm].values, both[bm].values)
                tests.append({"horizon": hz, "model": mlm, "baseline": bm, "n": len(both),
                              "model_right_baseline_wrong": n01, "baseline_right_model_wrong": n10, "mcnemar_p": p,
                              "model_accuracy": float((both[mlm].values == yy).mean()),
                              "baseline_accuracy": float((both[bm].values == yy).mean())})
    tests = pd.DataFrame(tests)

    kup = pd.DataFrame()
    if (RES / "var_backtest.parquet").exists():
        sys.path.insert(0, str(ROOT / "src"))
        for cand in (ROOT.parent / "risk_engine", ROOT.parent / "cse_risk_engine"):
            sys.path.insert(0, str(cand))
        from cse_risk_engine import kupiec_pof_test
        from models.universe import member_flag
        bt = pd.read_parquet(RES / "var_backtest.parquet")
        bt = bt[member_flag(bt.rename(columns={"date": "date"})).values]   # S&P SL20 member-days only
        rows = []
        for (sid, c), g in list(bt.groupby(["security_id", "confidence"])) + [(("POOLED", c), g) for c, g in bt.groupby("confidence")]:
            k = kupiec_pof_test(g["breach"].values, confidence=c).as_dict()
            k.update({"security_id": sid, "confidence": c})
            rows.append(k)
        kup = pd.DataFrame(rows)
        print(kup[kup["security_id"] == "POOLED"][["confidence", "T", "x", "observed_rate", "expected_rate", "LR_POF", "p_value", "verdict"]].to_string(index=False))
        ks = kup[kup["security_id"] != "POOLED"]
        print(ks.groupby("confidence").agg(stocks=("security_id", "count"), passed=("passed", "sum"), mean_rate=("observed_rate", "mean")).to_string())
    glob = pd.read_parquet(RES / "xgb_shap_global.parquet") if (RES / "xgb_shap_global.parquet").exists() else pd.DataFrame()
    gsum = glob.groupby(["horizon", "feature"])["mean_abs_shap"].mean().reset_index().sort_values(["horizon", "mean_abs_shap"], ascending=[True, False]) if len(glob) else glob
    conf_rows = []
    for (m, hz), g in ml[ml["label"].notna()].groupby(["model", "horizon"]):
        dist = (g["p_up"] - g["threshold"]).abs()
        for q in (0.0, 0.05, 0.10, 0.15):
            s_ = g[dist >= q]
            conf_rows.append({"model": m, "horizon": hz, "min_distance_from_threshold": q,
                              "coverage": len(s_) / len(g), "n": len(s_),
                              "accuracy": float((s_["pred"] == s_["label"]).mean()) if len(s_) else np.nan})
    with pd.ExcelWriter(RES / "evaluation.xlsx") as xw:
        pd.DataFrame(conf_rows).to_excel(xw, sheet_name="Accuracy by confidence", index=False)
        overall.sort_values(["horizon", "f1_macro"], ascending=[True, False]).to_excel(xw, sheet_name="Overall", index=False)
        by_block.to_excel(xw, sheet_name="By block", index=False)
        by_stock.to_excel(xw, sheet_name="By stock", index=False)
        tests.to_excel(xw, sheet_name="McNemar vs baselines", index=False)
        if len(kup):
            kup.to_excel(xw, sheet_name="Kupiec", index=False)
        if len(gsum):
            gsum.to_excel(xw, sheet_name="SHAP global (XGB)", index=False)
    cols = ["model", "horizon", "n", "accuracy", "balanced_accuracy", "f1_macro", "auc", "binom_p_vs_50pct"]
    print(overall[cols].sort_values(["horizon", "f1_macro"], ascending=[True, False]).round(4).to_string(index=False))
    if len(tests):
        print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
