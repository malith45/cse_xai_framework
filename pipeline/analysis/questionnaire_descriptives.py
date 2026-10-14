"""Descriptive statistics for the requirements questionnaire (project file 08).

Single-choice items: frequency and % of respondents. Multi-select items (Q5, Q13, Q14, Q19, Q22):
each option's % of respondents who ticked it (so percentages can sum to more than 100%).
Likert items (Q8, Q11, Q16, Q21) are also summarised as median and % agree (Agree + Strongly agree).
Open-ended Q24 is listed verbatim for thematic coding by the author (no automated interpretation).

Usage: python analysis/questionnaire_descriptives.py <csv> <out.xlsx>
"""

import re
import sys

import pandas as pd

MULTI = ("5.", "13.", "14.", "19.", "22.")
LIKERT = {"Strongly disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly Agree": 5, "Strongly agree": 5}


def short(c: str) -> str:
    return " ".join(c.split())[:90]


def main(src, dst):
    df = pd.read_csv(src)
    n = len(df)
    tables, summary = [], []
    for c in df.columns[1:]:
        qn = c.split(".")[0] + "."
        if c.startswith("24."):
            continue
        s = df[c].dropna().astype(str)
        if c.startswith(MULTI):
            opts = s.str.split(";").explode().str.strip()
            t = opts.value_counts().rename("count").to_frame()
        else:
            t = s.value_counts().rename("count").to_frame()
        t["% of respondents"] = (100 * t["count"] / n).round(1)
        t.insert(0, "question", short(c))
        t.index.name = "answer"
        tables.append(t.reset_index())
        if s.isin(LIKERT).all() and len(s):
            v = s.map(LIKERT)
            summary.append({"question": short(c), "n": len(v), "median (1-5)": v.median(), "mean (1-5)": round(v.mean(), 2),
                            "% agree or strongly agree": round(100 * (v >= 4).mean(), 1),
                            "% disagree or strongly disagree": round(100 * (v <= 2).mean(), 1)})
    open_ended = df[[c for c in df.columns if c.startswith("24.")]].dropna()
    open_ended.columns = ["Q24 response (verbatim)"]
    with pd.ExcelWriter(dst) as xw:
        pd.DataFrame({"item": ["respondents", "first response", "last response"],
                      "value": [n, df["Timestamp"].min(), df["Timestamp"].max()]}).to_excel(xw, sheet_name="Overview", index=False)
        pd.concat(tables).to_excel(xw, sheet_name="Frequencies", index=False)
        pd.DataFrame(summary).to_excel(xw, sheet_name="Likert summary", index=False)
        open_ended.to_excel(xw, sheet_name="Q24 open-ended", index=False)
    print(f"{n} responses -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
