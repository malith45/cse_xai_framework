"""Run the LSTM/GRU walk-forward (M4) and expected-gradient SHAP (M5). Designed for a Kaggle GPU
notebook (Table 4.8) but runs anywhere; uses CUDA when available.

Inputs : <data>/panel_universe.parquet (written by scripts/export_for_kaggle.py)
Outputs: <out>/rnn_predictions.parquet, rnn_shap_top10.parquet, rnn_shap_global.parquet,
         registry_rnn.json, rnn_selection.json (decision D5)

Usage: python scripts/run_rnn.py --data /kaggle/input/cse-panel --out /kaggle/working [--trials 15]
       [--archs lstm gru] [--horizons t1 t5] [--shap-nsamples 50]
Checkpointing: each finished (arch, horizon, block) is saved to <out>/parts/, and a restarted run
skips parts that already exist, so a session that hits the 12-hour limit can simply be re-run.
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
from models.explain import narrative, top_k_long  # noqa: E402
from models.rnn import DEVICE, gradient_shap, predict_rnn, train_rnn_block  # noqa: E402
from models.walkforward import make_blocks  # noqa: E402
from pipeline.features import FEATURES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=C.RNN_TRIALS)
    ap.add_argument("--archs", nargs="+", default=["lstm", "gru"])
    ap.add_argument("--horizons", nargs="+", default=list(C.HORIZONS))
    ap.add_argument("--shap-nsamples", type=int, default=50)
    ap.add_argument("--blocks", type=int, nargs="*", default=None, help="smoke tests only")
    a = ap.parse_args()
    out = Path(a.out)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(Path(a.data) / "panel_universe.parquet")
    names = panel.groupby("security_id")["short_name"].last() if "short_name" in panel else {}
    n_days = int(panel["n_calendar_days"].iloc[0]) if "n_calendar_days" in panel else int(panel["cal_pos"].max()) + 1
    blocks = make_blocks(n_days)
    print(f"device={DEVICE}; rows={len(panel)}; blocks={len(blocks)}", flush=True)

    for arch in a.archs:
        for hz in a.horizons:
            label = C.HORIZONS[hz]
            for blk in blocks:
                if a.blocks is not None and blk.b not in a.blocks:
                    continue
                tag = f"{arch}_{hz}_b{blk.b:02d}"
                if (parts / f"{tag}_pred.parquet").exists():
                    continue
                t0 = time.time()
                res = train_rnn_block(panel, FEATURES, label, blk, arch, n_trials=a.trials)
                prob, X, k = predict_rnn(res, panel, FEATURES, label, blk)
                p = k[["security_id", "date", "cal_pos", label]].rename(columns={label: "label"}).copy()
                p["horizon"], p["block"], p["model"] = hz, blk.b, arch
                p["p_up"], p["threshold"] = prob, res.threshold
                p["pred"] = (prob >= res.threshold).astype(int)
                p.to_parquet(parts / f"{tag}_pred.parquet")
                reg = {"model": arch, "horizon": hz, "block": blk.b, "train": blk.train,
                       "inner_val": blk.inner_val, "test": blk.test, "threshold": res.threshold,
                       "inner_val_f1_macro": res.inner_val_score, "params": res.params,
                       "seconds": round(time.time() - t0, 1)}
                json.dump(reg, open(parts / f"{tag}_reg.json", "w"), default=str)
                # attributions for this block (kept for both archs; only the deployed one is published)
                phi = gradient_shap(res, X, nsamples=a.shap_nsamples)
                base = np.zeros(len(phi))
                raw = k[FEATURES].copy()
                top = top_k_long(phi, base, raw, FEATURES, k[["security_id", "date"]])
                top["horizon"], top["block"], top["model"] = hz, blk.b, arch
                top.to_parquet(parts / f"{tag}_shap.parquet")
                g = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(phi).mean(axis=0),
                                  "horizon": hz, "block": blk.b, "model": arch})
                g.to_parquet(parts / f"{tag}_glob.parquet")
                print(f"{tag}: inner-val F1m={res.inner_val_score:.3f} ({time.time() - t0:.0f}s)", flush=True)

    # combine and apply decision D5 (mean inner-validation score across blocks, per horizon)
    regs = [json.load(open(f)) for f in sorted(parts.glob("*_reg.json"))]
    reg = pd.DataFrame(regs)
    sel = reg.groupby(["horizon", "model"])["inner_val_f1_macro"].mean().unstack()
    choice = {hz: sel.loc[hz].idxmax() for hz in sel.index}
    preds = pd.concat(pd.read_parquet(f) for f in sorted(parts.glob("*_pred.parquet")))
    shap_all = pd.concat(pd.read_parquet(f) for f in sorted(parts.glob("*_shap.parquet")))
    glob = pd.concat(pd.read_parquet(f) for f in sorted(parts.glob("*_glob.parquet")))
    preds.to_parquet(out / "rnn_predictions.parquet")
    shap_all.to_parquet(out / "rnn_shap_top10.parquet")
    glob.to_parquet(out / "rnn_shap_global.parquet")
    json.dump(regs, open(out / "registry_rnn.json", "w"), indent=1, default=str)
    json.dump({"mean_inner_val_f1_macro": sel.to_dict(), "deployed": choice}, open(out / "rnn_selection.json", "w"), indent=1)
    print("D5 selection:", choice, flush=True)


if __name__ == "__main__":
    main()
