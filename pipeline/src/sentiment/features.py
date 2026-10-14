"""M3 aggregation -> M2 sentiment features (Chapter 4 §4.6.4, Table 4.6 "company and market scores,
article counts, 3-day decay").

Input: articles_scored.parquet from the Kaggle run (one row per article: score s = P(pos) - P(neg),
published time, linked companies "A-N-0000|B-N-0000", market_news flag). Every article is first
assigned to a trading day with the 14:30 rule (pipeline step 10, ``assign_trading_day``): the
features on day t use only articles counted for day t or earlier, i.e. published before the close
of t.

Features (all NaN before the news period starts, which XGBoost handles natively; the RNN export
fills scores with 0 and counts with 0):
  sent_co_mean    mean score of the company's articles on day t
  sent_co_count   number of the company's articles on day t (0 when none)
  sent_co_decay3  decayed company score over days t, t-1, t-2 (weights DECAY_WEIGHTS; days without
                  articles are skipped and the weights renormalised)
  sent_mkt_mean   market-day index: mean score of all articles that are market news or linked to
                  any constituent on day t (the fallback when a company has no news)
  sent_mkt_count  number of articles in that index on day t
  sent_mkt_decay3 decayed market index, same weights
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentiment.text import assign_trading_day

DECAY_WEIGHTS = (1.0, 0.5, 0.25)   # day t, t-1, t-2: halving weights (author's decision 2026-10-14)
SENTIMENT = ["sent_co_mean", "sent_co_count", "sent_co_decay3", "sent_mkt_mean", "sent_mkt_count", "sent_mkt_decay3"]


def _decay(mean: pd.DataFrame, count: pd.DataFrame) -> pd.DataFrame:
    """mean/count: trading-day x entity matrices (count 0 where no news). Weighted by weight x count."""
    num = pd.DataFrame(0.0, index=mean.index, columns=mean.columns)
    den = pd.DataFrame(0.0, index=mean.index, columns=mean.columns)
    for k, w in enumerate(DECAY_WEIGHTS):
        m = mean.shift(k).fillna(0.0)
        c = count.shift(k).fillna(0.0)
        num += w * m * c
        den += w * c
    return (num / den.replace(0.0, np.nan))


def sentiment_features(articles: pd.DataFrame, calendar: pd.DatetimeIndex, securities: list[str],
                       start: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Long table (security_id, date) x SENTIMENT for every trading day from ``start`` (first news day
    if None) to the end of the calendar. Days before ``start`` are absent (NaN after merge)."""
    cal = pd.DatetimeIndex(sorted(calendar))
    a = articles.reset_index(drop=True).copy()
    a["trading_day"] = assign_trading_day(a["published_at"] if "published_at" in a else a["published"], cal)
    a = a[a["trading_day"].notna() & a["score"].notna()]
    start = pd.Timestamp(start) if start is not None else a["trading_day"].min()
    days = cal[cal >= start]

    # company-day
    links = a.assign(security_id=a["companies"].fillna("").str.split("|")).explode("security_id")
    links = links[links["security_id"].isin(securities)]
    g = links.groupby(["trading_day", "security_id"])["score"]
    co_mean = g.mean().unstack().reindex(index=days, columns=securities)
    co_count = g.size().unstack().reindex(index=days, columns=securities).fillna(0.0)
    co_decay = _decay(co_mean, co_count)

    # market-day index: market news or any linked constituent
    mk = a[a["market_news"].fillna(False).astype(bool) | a.index.isin(links.index)]
    mg = mk.groupby("trading_day")["score"]
    mk_mean = mg.mean().reindex(days).to_frame("m")
    mk_count = mg.size().reindex(days).fillna(0.0).to_frame("m")
    mk_decay = _decay(mk_mean, mk_count)

    out = pd.DataFrame({
        "sent_co_mean": co_mean.stack(future_stack=True),
        "sent_co_count": co_count.stack(future_stack=True),
        "sent_co_decay3": co_decay.stack(future_stack=True),
    })
    out.index.names = ["date", "security_id"]
    out = out.reset_index()
    mkt = pd.DataFrame({"date": days, "sent_mkt_mean": mk_mean["m"].values, "sent_mkt_count": mk_count["m"].values,
                        "sent_mkt_decay3": mk_decay["m"].values})
    return out.merge(mkt, on="date", how="left")[["security_id", "date", *SENTIMENT]]
