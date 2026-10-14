"""Objective O4 analysis: System Usability Scale (Brooke, 1996) and the XAI Explanation Satisfaction
and Trust scales (Hoffman et al., 2023) from the user-evaluation Google Form export.

Expected columns (as named in the form): SUS1..SUS10, ES1..ES8, TR1..TR8 (1 = Strongly disagree ...
5 = Strongly agree), plus task-success columns TASK1..TASK5 (Yes/No) and profile questions.

Scoring
* SUS: odd items contribute (x - 1), even items (5 - x); sum x 2.5 -> 0-100. Benchmarks: 68 = average
  (Sauro & Lewis, 2016); grade bands A+ ... F from Sauro & Lewis's curved grading scale.
* ES and TR: item mean on 1-5; TR6 ("I am wary of the tool") is reverse-scored.
* Reliability: Cronbach's alpha per scale (internal consistency AFTER collection, as the course
  director requires), with item-total correlations.

Usage: python analysis/user_evaluation.py <form_export.csv> <out.xlsx>
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

LIKERT = {"strongly disagree": 1, "disagree": 2, "neutral": 3, "agree": 4, "strongly agree": 5}
SUS_GRADES = [(84.1, "A+"), (80.8, "A"), (78.9, "A-"), (77.2, "B+"), (74.1, "B"), (72.6, "B-"),
              (71.1, "C+"), (65.0, "C"), (62.7, "C-"), (51.7, "D"), (0, "F")]


def to_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return s.astype(str).str.strip().str.lower().map(lambda v: LIKERT.get(v, pd.to_numeric(re.sub(r"\D", "", v) or "nan", errors="coerce")))


def cronbach_alpha(items: pd.DataFrame) -> float:
    items = items.dropna()
    k = items.shape[1]
    if k < 2 or len(items) < 3:
        return np.nan
    return float(k / (k - 1) * (1 - items.var(ddof=1).sum() / items.sum(axis=1).var(ddof=1)))


def pick(df: pd.DataFrame, prefix: str, n: int) -> pd.DataFrame:
    cols = []
    for i in range(1, n + 1):
        c = next((c for c in df.columns if re.match(rf"^\s*{prefix}\s*{i}\b", c, re.I)), None)
        if c is None:
            raise KeyError(f"column {prefix}{i} not found")
        cols.append(c)
    out = df[cols].apply(to_num)
    out.columns = [f"{prefix}{i}" for i in range(1, n + 1)]
    return out


def main(src: str, dst: str):
    df = pd.read_csv(src)
    sus = pick(df, "SUS", 10)
    es = pick(df, "ES", 8)
    tr = pick(df, "TR", 8)
    tr_adj = tr.copy()
    tr_adj["TR6"] = 6 - tr_adj["TR6"]
    sus_adj = sus.copy()
    for i in range(1, 11):
        sus_adj[f"SUS{i}"] = sus[f"SUS{i}"] - 1 if i % 2 else 5 - sus[f"SUS{i}"]
    score = sus_adj.sum(axis=1) * 2.5
    per = pd.DataFrame({"SUS score": score, "Explanation satisfaction (1-5)": es.mean(axis=1),
                        "Trust (1-5)": tr_adj.mean(axis=1)})

    def ci(x):
        x = x.dropna()
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
        return x.mean(), x.mean() - h, x.mean() + h

    rows = []
    for name, col, items in (("SUS", "SUS score", sus_adj), ("Explanation satisfaction", "Explanation satisfaction (1-5)", es),
                             ("Trust", "Trust (1-5)", tr_adj)):
        m, lo, hi = ci(per[col])
        rows.append({"scale": name, "n": int(per[col].notna().sum()), "mean": m, "95% CI low": lo, "95% CI high": hi,
                     "sd": per[col].std(ddof=1), "median": per[col].median(), "cronbach_alpha": cronbach_alpha(items)})
    summary = pd.DataFrame(rows)
    mean_sus = per["SUS score"].mean()
    grade = next(g for t, g in SUS_GRADES if mean_sus >= t)
    one_sample = stats.ttest_1samp(per["SUS score"].dropna(), 68.0)
    summary.loc[summary["scale"] == "SUS", "grade (Sauro & Lewis)"] = grade
    summary.loc[summary["scale"] == "SUS", "t vs 68"] = one_sample.statistic
    summary.loc[summary["scale"] == "SUS", "p vs 68"] = one_sample.pvalue

    item_stats = []
    for prefix, items in (("SUS", sus), ("ES", es), ("TR", tr)):
        adj = {"SUS": sus_adj, "ES": es, "TR": tr_adj}[prefix]
        total = adj.sum(axis=1)
        for c in items.columns:
            item_stats.append({"item": c, "mean (raw 1-5)": items[c].mean(), "sd": items[c].std(ddof=1),
                               "corrected item-total r": adj[c].corr(total - adj[c])})
    tasks = [c for c in df.columns if re.match(r"^\s*TASK\s*\d", c, re.I)]
    task_tbl = pd.DataFrame({"task": tasks, "completed (%)": [100 * df[c].astype(str).str.lower().str.startswith("y").mean() for c in tasks]})
    with pd.ExcelWriter(dst) as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(item_stats).to_excel(xw, sheet_name="Items", index=False)
        task_tbl.to_excel(xw, sheet_name="Tasks", index=False)
        per.to_excel(xw, sheet_name="Per respondent", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
