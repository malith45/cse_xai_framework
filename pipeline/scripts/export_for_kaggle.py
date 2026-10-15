"""Write the universe panel (derived features + labels only; no raw licensed price rows) and the
code needed by scripts/run_rnn.py into dist/kaggle/ for upload as a PRIVATE Kaggle dataset.

Usage: python scripts/export_for_kaggle.py [--feature-set prices|macro|full]
The chosen feature list is written to features.json beside the panel; run_rnn.py reads it."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models.universe import load_universe, load_universe_full  # noqa: E402
from pipeline.features import FEATURE_SETS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--feature-set", default="full" if "full" in FEATURE_SETS else "macro", choices=sorted(FEATURE_SETS))
ap.add_argument("--only", choices=["rnn", "sentiment", "both"], default="both")
ARGS = ap.parse_args()
FEATURES = FEATURE_SETS[ARGS.feature_set]



def export_rnn():
    dst = ROOT / "dist" / "kaggle" / "rnn_input"
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "code").mkdir(parents=True)
    pa = pd.read_parquet(ROOT / "work" / "panel_all.parquet")
    missing = [f for f in FEATURES if f not in pa.columns]
    if missing:
        sys.exit(f"panel_all.parquet lacks {missing}: run build_sentiment.py and build_panel.py first")
    _, info = load_universe(pa)
    panel = load_universe_full(pa)
    names = pd.read_parquet(ROOT / "work" / "prices_adj.parquet").groupby("security_id")["short_name"].last()
    keep = ["security_id", "date", "cal_pos", "member", "y1", "y5"] + FEATURES
    out = panel[keep].copy()
    out["short_name"] = out["security_id"].map(names)
    out["n_calendar_days"] = int(pa["cal_pos"].max()) + 1
    out.to_parquet(dst / "panel_universe.parquet")
    json.dump(FEATURES, open(dst / "features.json", "w"))
    for d in ("src/models", "src/pipeline", "src/sentiment"):   # pipeline.features imports sentiment.features
        shutil.copytree(ROOT / d, dst / "code" / d, ignore=shutil.ignore_patterns("__pycache__"))
    (dst / "code" / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "run_rnn.py", dst / "code" / "scripts" / "run_rnn.py")
    (dst / "UNIVERSE.txt").write_text(info + "\n")
    print(info, out.shape, len(FEATURES), "features ->", dst)


def export_sentiment_code():
    """Sentiment code package (no data)."""
    sdst = ROOT / "dist" / "kaggle" / "sentiment_code"
    if sdst.exists():
        shutil.rmtree(sdst)
    shutil.copytree(ROOT / "src" / "sentiment", sdst / "code" / "src" / "sentiment",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (sdst / "code" / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "kaggle_sentiment.py", sdst / "code" / "scripts" / "kaggle_sentiment.py")
    print("sentiment code ->", sdst)


if ARGS.only in ("rnn", "both"):
    export_rnn()
if ARGS.only in ("sentiment", "both"):
    export_sentiment_code()
