"""Choose the deployed XGBoost feature set per horizon by inner-validation macro-F1 (never test blocks).

Same rule as decision D5 (LSTM vs GRU): for each horizon, the feature set with the higher mean
inner-validation macro-F1 across the walk-forward blocks is deployed; a tie keeps the simpler set.
Reads work/results/ablation/<set>/registry_xgb.json, writes work/results/feature_selection.json and
assembles the deployed xgb_* files in work/results/ from the selected set of each horizon.

Usage: python scripts/select_feature_set.py [--sets prices macro full]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "work" / "results"
ORDER = ["prices", "macro", "full"]          # simplest first (tie-break)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=None)
    a = ap.parse_args()
    sets = [s for s in (a.sets or ORDER) if (RES / "ablation" / s / "registry_xgb.json").exists()]
    rows = []
    for s in sets:
        for r in json.load(open(RES / "ablation" / s / "registry_xgb.json")):
            rows.append({"set": s, "horizon": r["horizon"], "block": r["block"], "iv": r["inner_val_f1_macro"]})
    d = pd.DataFrame(rows)
    means = d.groupby(["horizon", "set"])["iv"].mean().unstack()[sets]
    chosen = {}
    for hz, row in means.iterrows():
        best = max(sets, key=lambda s: (round(row[s], 10), -sets.index(s)))
        chosen[hz] = best
    sel = {"rule": "highest mean inner-validation macro-F1 per horizon; tie -> simpler set",
           "mean_inner_val_f1_macro": {hz: {s: float(means.loc[hz, s]) for s in sets} for hz in means.index},
           "deployed": chosen}
    json.dump(sel, open(RES / "feature_selection.json", "w"), indent=1)

    pred, shap, glob, reg = [], [], [], []
    for hz, s in chosen.items():
        src = RES / "ablation" / s
        p = pd.read_parquet(src / "xgb_predictions.parquet")
        pred.append(p[p["horizon"] == hz].assign(feature_set=s))
        t = pd.read_parquet(src / "xgb_shap_top10.parquet")
        shap.append(t[t["horizon"] == hz])
        g = pd.read_parquet(src / "xgb_shap_global.parquet")
        glob.append(g[g["horizon"] == hz])
        reg += [r for r in json.load(open(src / "registry_xgb.json")) if r["horizon"] == hz]
    pd.concat(pred, ignore_index=True).to_parquet(RES / "xgb_predictions.parquet")
    pd.concat(shap, ignore_index=True).to_parquet(RES / "xgb_shap_top10.parquet")
    pd.concat(glob, ignore_index=True).to_parquet(RES / "xgb_shap_global.parquet")
    json.dump(reg, open(RES / "registry_xgb.json", "w"), indent=1, default=str)
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
