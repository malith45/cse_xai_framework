"""M3 -> M2/M8: turn the Kaggle article scores into sentiment features and dashboard tables.

Input : work/results/articles_scored.parquet (from scripts/kaggle_sentiment.py on Kaggle)
Output: work/sentiment_features.parquet   (security_id, date, SENTIMENT) - merged by build_panel.py
        work/results/sentiment_daily.parquet (dashboard: company and market score per trading day)
        work/results/article.parquet         (dashboard: headline, source, date, link, score - never body text)
        work/results/sentiment_summary.json  (counts for Chapter 5)

Usage: python scripts/build_sentiment.py && python scripts/build_panel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.prices import STUDY_START  # noqa: E402
from sentiment.features import SENTIMENT, sentiment_features  # noqa: E402
from sentiment.text import assign_trading_day  # noqa: E402

WORK, RES = ROOT / "work", ROOT / "work" / "results"
if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":      # test the plumbing without touching work/
    WORK = RES = Path(sys.argv[2])


def main():
    arts = pd.read_parquet(RES / "articles_scored.parquet")
    src = ROOT / "work"
    ix = pd.read_parquet(src / "indices.parquet")
    cal = pd.DatetimeIndex(sorted(ix["date"].unique()))
    sids = sorted(pd.read_parquet(src / "prices_adj.parquet", columns=["security_id"])["security_id"].unique())
    if "published_at" not in arts or arts["published_at"].isna().all():
        arts["published_at"] = arts["published"]
    # the article-page time (when present) is more precise than the sitemap date
    arts["published_at"] = arts["published"].where(arts["published"].fillna("").str.contains(r"\d{1,2}:\d{2}"),
                                                   arts["published_at"])
    feats = sentiment_features(arts, cal, sids, start=STUDY_START)
    feats.to_parquet(WORK / "sentiment_features.parquet")

    # dashboard: company-day and market-day series
    co = feats[feats["sent_co_count"] > 0][["security_id", "date", "sent_co_mean", "sent_co_count"]]
    co = co.rename(columns={"sent_co_mean": "company_score", "sent_co_count": "article_count"})
    mk = feats.drop_duplicates("date")[["date", "sent_mkt_mean", "sent_mkt_count"]]
    co = co.merge(mk, on="date", how="left").rename(columns={"sent_mkt_mean": "market_score", "sent_mkt_count": "market_articles"})
    mrow = mk[mk["sent_mkt_count"] > 0].rename(columns={"sent_mkt_mean": "market_score", "sent_mkt_count": "market_articles"})
    mrow = mrow.assign(security_id="_MARKET", company_score=np.nan, article_count=0)
    daily = pd.concat([co, mrow[co.columns]], ignore_index=True)
    daily.to_parquet(RES / "sentiment_daily.parquet")

    # dashboard: one row per article-company link (news that names no listed company keeps security_id
    # empty and appears in the market-wide list); never body text
    a = arts.copy()
    a["trading_day"] = assign_trading_day(a["published_at"], cal)
    a = a[a["trading_day"].notna()]
    a["security_id"] = a["companies"].fillna("").str.split("|")
    a = a.explode("security_id")
    a["security_id"] = a["security_id"].replace("", None)
    art = pd.DataFrame({"security_id": a["security_id"], "trading_day": a["trading_day"],
                        "published": a["published_at"].astype(str).str[:19], "headline": a["title"].fillna(""),
                        "source": a["source"], "language": a["lang"], "url": a["url"], "score": a["score"].round(3)})
    art.to_parquet(RES / "article.parquet")

    summary = {
        "articles_scored": int(len(arts)),
        "by_language": arts["lang"].value_counts().to_dict(),
        "by_source": arts["source"].value_counts().to_dict(),
        "fallback_used": int(arts["fallback_used"].fillna(False).sum()) if "fallback_used" in arts else None,
        "linked_to_a_company": int((arts["companies"].fillna("") != "").sum()),
        "market_news": int(arts["market_news"].fillna(False).astype(bool).sum()),
        "company_days_with_news": int((feats["sent_co_count"] > 0).sum()),
        "trading_days_with_market_news": int((mk["sent_mkt_count"] > 0).sum()),
        "first_trading_day": str(feats["date"].min().date()), "last_trading_day": str(feats["date"].max().date()),
        "features": SENTIMENT,
    }
    json.dump(summary, open(RES / "sentiment_summary.json", "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
