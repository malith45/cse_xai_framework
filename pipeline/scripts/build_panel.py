"""M1 + M2: rebuild the adjusted price table and the point-in-time feature panel.

Usage:
    python scripts/build_panel.py            # reuse work/prices_adj.parquet + work/indices.parquet
    python scripts/build_panel.py --from-raw # re-read the CSE Data Library (needs CSE_DATA_ROOT)

Output: work/panel_all.parquet (price, index and macro features, labels y1/y5) and
work/m2_report.json. The licensed raw files stay outside the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.features import build_panel  # noqa: E402
from pipeline.macro import macro_features  # noqa: E402

WORK = ROOT / "work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-raw", action="store_true")
    a = ap.parse_args()
    if a.from_raw:
        from dataclasses import asdict
        from pipeline.corporate_actions import apply_adjustment, build_events
        from pipeline.market import load_indices
        from pipeline.prices import load_prices
        root = Path(os.environ["CSE_DATA_ROOT"])
        prices, rep = load_prices(root)
        events, unmatched = build_events(root, prices)
        adj = apply_adjustment(prices, events)
        ix = load_indices(root)
        WORK.mkdir(exist_ok=True)
        adj.to_parquet(WORK / "prices_adj.parquet")
        events.to_parquet(WORK / "corporate_events.parquet")
        ix.to_parquet(WORK / "indices.parquet")
        json.dump(asdict(rep) if hasattr(rep, "__dataclass_fields__") else rep, open(WORK / "m1_load_report.json", "w"),
                  indent=1, default=str)
    adj = pd.read_parquet(WORK / "prices_adj.parquet")
    ix = pd.read_parquet(WORK / "indices.parquet")
    cal = pd.DatetimeIndex(sorted(ix["date"].unique()))
    macro = macro_features(cal)
    panel, report = build_panel(adj, ix, macro=macro)
    sf = WORK / "sentiment_features.parquet"
    if sf.exists():   # M3 output (scripts/build_sentiment.py); company-day features, NaN before the news period
        s = pd.read_parquet(sf)
        panel = panel.merge(s, on=["security_id", "date"], how="left")
        report["sentiment_rows_with_company_news"] = int((panel["sent_co_count"] > 0).sum())
    panel.to_parquet(WORK / "panel_all.parquet")
    report["macro_columns"] = list(macro.columns)
    json.dump(report, open(WORK / "m2_report.json", "w"), indent=1)
    print(report)


if __name__ == "__main__":
    main()
