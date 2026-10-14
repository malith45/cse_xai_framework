"""Run M4 (XGBoost + baselines) and M5 (TreeSHAP + narratives) over every walk-forward block.

Usage: python scripts/run_xgb_baselines.py [--trials 30] [--skip-baselines]
Outputs (work/results/): xgb_predictions.parquet, xgb_shap_top10.parquet, xgb_shap_global.parquet,
baseline_predictions.parquet, registry_xgb.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import config as C  # noqa: E402
from models.baselines import model_baseline, naive  # noqa: E402
from models.explain import narrative, top_k_long, tree_shap  # noqa: E402
from models.walkforward import in_range, make_blocks, predict_xgb, train_xgb_block  # noqa: E402
from models.universe import load_universe, load_universe_full  # noqa: E402
from pipeline.features import FEATURES  # noqa: E402

OUT = ROOT / "work" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=C.XGB_TRIALS)
    ap.add_argument("--skip-baselines", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    panel_all = pd.read_parquet(ROOT / "work" / "panel_all.parquet")
    panel, uni_info = load_universe(panel_all)
    names = pd.read_parquet(ROOT / "work" / "prices_adj.parquet").groupby("security_id")["short_name"].last()
    n_days = int(panel_all["cal_pos"].max()) + 1
    blocks = make_blocks(n_days)
    cats = sorted(panel["security_id"].unique())
    print(f"universe: {uni_info}; blocks: {len(blocks)}; rows: {len(panel)}", flush=True)

    preds, shap_rows, glob, registry = [], [], [], []
    parts = OUT / "xgb_parts"
    parts.mkdir(exist_ok=True)
    for hz, label in C.HORIZONS.items():
        for blk in blocks:
            tag = f"{hz}_b{blk.b:02d}"
            if (parts / f"{tag}_pred.parquet").exists():
                preds.append(pd.read_parquet(parts / f"{tag}_pred.parquet"))
                shap_rows.append(pd.read_parquet(parts / f"{tag}_shap.parquet"))
                glob.append(pd.read_parquet(parts / f"{tag}_glob.parquet"))
                registry.append(json.load(open(parts / f"{tag}_reg.json")))
                continue
            t0 = time.time()
            res = train_xgb_block(panel, FEATURES, label, blk, cats, n_trials=a.trials)
            test = panel[in_range(panel["cal_pos"], blk.test)]
            prob, X = predict_xgb(res, test, cats)
            phi, base = tree_shap(res.model, X)
            cols = list(X.columns)
            keys = test[["security_id", "date"]]
            raw = test[FEATURES].copy()
            raw["stock"] = test["security_id"].values
            top = top_k_long(phi, base, raw, cols, keys)
            top["horizon"], top["block"] = hz, blk.b
            shap_rows.append(top)
            glob.append(pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(phi).mean(axis=0),
                                      "horizon": hz, "block": blk.b}))
            p = test[["security_id", "date", "cal_pos", label]].rename(columns={label: "label"}).copy()
            p["horizon"], p["block"], p["model"] = hz, blk.b, "xgboost"
            p["p_up"], p["threshold"] = prob, res.threshold
            p["pred"] = (prob >= res.threshold).astype(int)
            # narratives
            narr = []
            groups = {k: g for k, g in top.groupby(["security_id", "date"], sort=False)}
            for sid, d, pu in zip(p["security_id"], p["date"], p["p_up"]):
                t = groups.get((sid, d), top.iloc[:0])
                narr.append(narrative(str(names.get(sid, sid)).title(), float(pu), hz, t))
            p["narrative"] = narr
            preds.append(p)
            registry.append({"model": "xgboost", "horizon": hz, "block": blk.b, "train": blk.train,
                             "inner_val": blk.inner_val, "test": blk.test, "threshold": res.threshold,
                             "inner_val_f1_macro": res.inner_val_score, "params": res.params,
                             "n_train_rows": int((in_range(panel["cal_pos"], blk.train) & panel[label].notna()).sum()),
                             "seconds": round(time.time() - t0, 1)})
            preds[-1].to_parquet(parts / f"{tag}_pred.parquet")
            shap_rows[-1].to_parquet(parts / f"{tag}_shap.parquet")
            glob[-1].to_parquet(parts / f"{tag}_glob.parquet")
            json.dump(registry[-1], open(parts / f"{tag}_reg.json", "w"), default=str)
            print(f"xgb {hz} block {blk.b}: inner-val F1m={res.inner_val_score:.3f} thr={res.threshold:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    pd.concat(preds).to_parquet(OUT / "xgb_predictions.parquet")
    pd.concat(shap_rows).to_parquet(OUT / "xgb_shap_top10.parquet")
    pd.concat(glob).to_parquet(OUT / "xgb_shap_global.parquet")
    json.dump(registry, open(OUT / "registry_xgb.json", "w"), indent=1, default=str)

    if a.skip_baselines:
        return
    base = []
    nv = naive(panel)
    for blk in blocks:
        t = nv[in_range(nv["cal_pos"], blk.test)].copy()
        t["block"], t["model"] = blk.b, "naive"
        base.append(t)
    # fit on each security's full return history (including days before it joined the index);
    # forecasts are kept only for days on which it was a constituent
    full = load_universe_full(panel_all) if "point-in-time" in uni_info else panel
    member_keys = set(zip(panel["security_id"], panel["date"]))
    for kind in ("arima", "garch"):
        for blk in blocks:
            t0 = time.time()
            f = model_baseline(full, blk, kind)
            if len(f):
                f = f[[k in member_keys for k in zip(f["security_id"], f["date"])]]
            f["block"], f["model"] = blk.b, kind
            base.append(f)
            print(f"{kind} block {blk.b} ({time.time() - t0:.0f}s)", flush=True)
            pd.concat(base).to_parquet(OUT / "baseline_predictions.parquet")
    pd.concat(base).to_parquet(OUT / "baseline_predictions.parquet")


if __name__ == "__main__":
    main()
