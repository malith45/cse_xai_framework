"""Write the universe panel (derived features + labels only; no raw licensed price rows) and the
code needed by scripts/run_rnn.py into dist/kaggle/ for upload as a PRIVATE Kaggle dataset."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models.universe import load_universe, load_universe_full  # noqa: E402
from pipeline.features import FEATURES  # noqa: E402

dst = ROOT / "dist" / "kaggle" / "rnn_input"
if dst.exists():
    shutil.rmtree(dst)
(dst / "code").mkdir(parents=True)
pa = pd.read_parquet(ROOT / "work" / "panel_all.parquet")
_, info = load_universe(pa)
panel = load_universe_full(pa)
names = pd.read_parquet(ROOT / "work" / "prices_adj.parquet").groupby("security_id")["short_name"].last()
keep = ["security_id", "date", "cal_pos", "member", "y1", "y5"] + FEATURES
out = panel[keep].copy()
out["short_name"] = out["security_id"].map(names)
out["n_calendar_days"] = int(pa["cal_pos"].max()) + 1
out.to_parquet(dst / "panel_universe.parquet")
for d in ("src/models", "src/pipeline"):
    shutil.copytree(ROOT / d, dst / "code" / d, ignore=shutil.ignore_patterns("__pycache__"))
(dst / "code" / "scripts").mkdir()
shutil.copy(ROOT / "scripts" / "run_rnn.py", dst / "code" / "scripts" / "run_rnn.py")
(dst / "UNIVERSE.txt").write_text(info + "\n")
print(info, out.shape, "->", dst)

# sentiment code package (no data)
sdst = ROOT / "dist" / "kaggle" / "sentiment_code"
if sdst.exists():
    shutil.rmtree(sdst)
shutil.copytree(ROOT / "src" / "sentiment", sdst / "code" / "src" / "sentiment", ignore=shutil.ignore_patterns("__pycache__"))
(sdst / "code" / "scripts").mkdir(parents=True)
shutil.copy(ROOT / "scripts" / "kaggle_sentiment.py", sdst / "code" / "scripts" / "kaggle_sentiment.py")
print("sentiment code ->", sdst)
