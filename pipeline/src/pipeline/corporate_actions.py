"""M1 - corporate-action adjustment (Chapter 4 §4.7 step 4).

Prices in the CSE Data Library are unadjusted. Each event becomes a multiplicative price factor
``f`` (< 1 for splits, bonus issues, rights issues and cash dividends) applied to every price
*before* the event's ex-date (backward adjustment, so the latest prices equal traded prices).

==============  =============================================  =====================================
Event           Factor                                         Ex-date
==============  =============================================  =====================================
Split  o:n      o / n                                          EFFECTIVE DATE (file 05), confirmed
                                                               against the price jump (+/-5 days)
Bonus  o:n*     o / (o + n)                                    estimated from the price jump within
                                                               [allotment - 20, allotment + 5]
Rights n:o**    (o*P + n*S) / ((o + n)*P), P = cum close        as bonus
Cash dividend   1 - D / P_cum (file 01 CUM PRICE)              DATE OF EX (file 01)
==============  =============================================  =====================================

\\* Bonus file: "Old Proportion 17.53, New Proportion 1" = 1 new share per 17.53 held.
\\** Rights "Propotion 5:24" = 5 new shares per 24 held, subscription price S.

Scrip dividends (file 02) are NOT adjusted separately: the dividend file's RATE OF DIVIDEND already
contains the scrip portion and the scrip allotment date equals the dividend's ex-date (checked on
DFCC, COMB, NDB 2025), so adjusting both would double count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from identifiers import parse
from readers import read_workbook

YEARS = [str(y) for y in range(2020, 2026)]


def _year_sheets(path: Path) -> list[str]:
    from readers.formats import list_sheet_names
    return [n for n in list_sheet_names(path) if (y := re.search(r"(20\d\d)", str(n))) and y.group(1) in YEARS]


def _sheets(path: Path, cols, required):
    wb = read_workbook(path, cols, required=required, sheets=_year_sheets(path))
    out = {}
    for name, df in wb.items():
        y = re.search(r"(20\d\d)", str(name))
        if y and y.group(1) in YEARS:
            out[y.group(1)] = df
    return out


def _date(v):
    if v is None or (isinstance(v, float) and v != v) or v is pd.NaT:
        return pd.NaT
    if isinstance(v, pd.Timestamp):
        return v.normalize()
    if hasattr(v, "year"):
        return pd.Timestamp(v).normalize()
    s = str(v).strip()
    for fmt in ("%d-%b-%y", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            pass
    return pd.NaT


def _num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return np.nan


# --------------------------------------------------------------------------- name -> security map

_DROP = re.compile(r"\b(PLC|LIMITED|LTD|COMPANY|CO|THE|OF|AND|HOLDINGS?|ORDINARY|SHARES?)\b")


def norm_name(s: str) -> str:
    s = str(s).upper().replace("&", " AND ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[-*].*?(NON[\s-]*VOTING|VOTING).*$", " ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = _DROP.sub(" ", s)
    return " ".join(s.split())


def share_class_from_name(s: str) -> str:
    u = str(s).upper()
    return "X" if re.search(r"NON[\s-]*VOTING", u) else "N"


def name_map(root: Path) -> dict[str, str]:
    """normalised company name -> company code, from the quoted-securities master (file 16)."""
    wb = _sheets(root / "16 List of Quoted Securities And Issued Quantity.xls",
                 ["security_id", "company_name"], ["security_id", "company_name"])
    m: dict[str, str] = {}
    for df in wb.values():
        for sid, name in zip(df["security_id"], df["company_name"]):
            try:
                p = parse(str(sid))
            except Exception:
                continue
            code = getattr(p, "company_code", None)
            if code and getattr(p, "share_class", None) == "N":
                m.setdefault(norm_name(name), code)
    return m


# Misspellings in files 03/04 that the quoted-securities master cannot resolve (checked by hand).
NAME_ALIASES = {
    "NDB BANK": "NDB", "LANKEM CEYON": "LCEY", "ASSOCIATE MOTOR FINANCE": "AMF",
    "LAXAPANA BATERIES": "LITE", "BALANGODA PLANATATIONS": "BALA", "BERUWELA REOSRTS": "BERU",
    "SMB LEASING": "SEMB",
}


def lookup_code(name: str, m: dict[str, str]) -> str | None:
    key = norm_name(name)
    if key in NAME_ALIASES:
        return NAME_ALIASES[key]
    if key in m:
        return m[key]
    # fall back to a unique prefix match on the first two words
    head = " ".join(key.split()[:2])
    hits = {c for k, c in m.items() if k.startswith(head)} if head else set()
    return hits.pop() if len(hits) == 1 else None


# --------------------------------------------------------------------------- event tables

def load_splits(root: Path) -> pd.DataFrame:
    cols = ["company_code", "company_name", "old_proportion", "new_proportion", "effective_date"]
    rows = []
    for y, df in _sheets(root / "05Sub Division (Share Splits).xls", cols, ["company_code", "effective_date"]).items():
        for _, r in df.iterrows():
            raw = str(r["company_code"]).strip()
            code = re.split(r"[.\-\s]", raw)[0].upper()
            name = str(r.get("company_name", ""))
            classes = ["N", "X"] if "N & X" in name.upper() else (["X"] if "-X-" in name.upper() else ["N"])
            o, n = _num(r["old_proportion"]), _num(r["new_proportion"])
            d = _date(r["effective_date"])
            if not code or np.isnan(o) or np.isnan(n) or pd.isna(d):
                continue
            for cls in classes:
                rows.append({"security_id": f"{code}-{cls}-0000", "type": "split", "ref_date": d,
                             "factor": o / n, "detail": f"{int(o)}:{int(n)}"})
    return pd.DataFrame(rows)


def load_dividends(root: Path) -> pd.DataFrame:
    cols = ["security_id", "rate_of_dividend", "date_of_ex", "cum_price", "ex_price"]
    rows = []
    for y, df in _sheets(root / "01Dividends.xls", cols, ["security_id", "date_of_ex"]).items():
        for _, r in df.iterrows():
            try:
                p = parse(str(r["security_id"]))
                sid = str(p)
            except Exception:
                continue
            d, cum = _date(r["date_of_ex"]), _num(r.get("cum_price"))
            rate = _num(r.get("rate_of_dividend"))
            if pd.isna(d) or np.isnan(rate) or np.isnan(cum) or cum <= 0 or rate <= 0 or rate >= cum:
                continue
            rows.append({"security_id": sid, "type": "dividend", "ref_date": d,
                         "factor": 1 - rate / cum, "detail": f"Rs {rate:g} on cum {cum:g}"})
    return pd.DataFrame(rows)


def load_bonus(root: Path, m: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Bonus issues. Column sets differ by year: either Old/New Proportion (old held : new given)
    or a single "Propotion" string such as ``25:01`` (read as old:new, i.e. 1 new per 25 held)."""
    path = root / "03Capitalization of Reserves (Bonus Issues).xls"
    cols = ["company_name"]
    wb = read_workbook(path, cols, required=["company_name"], sheets=_year_sheets(path))
    rows, unmatched = [], []
    for name, df in wb.items():
        y = re.search(r"(20\d\d)", str(name)).group(1)
        if df.empty:
            continue
        date_col = next((c for c in ("allotment_date", "date_of_allotment") if c in df.columns), None)
        for _, r in df.iterrows():
            nm = r["company_name"]
            if not isinstance(nm, str) or not nm.strip():
                continue
            if "old_proportion" in df.columns:
                o, n = _num(r["old_proportion"]), _num(r["new_proportion"])
            elif "propotion" in df.columns:
                mm = re.match(r"^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$", str(r["propotion"]))
                o, n = (float(mm.group(1)), float(mm.group(2))) if mm else (np.nan, np.nan)
            else:
                o = n = np.nan
            d = _date(r[date_col]) if date_col else pd.NaT
            code = lookup_code(nm, m)
            if code is None:
                unmatched.append(f"bonus {y}: {nm}")
                continue
            if np.isnan(o) or np.isnan(n) or pd.isna(d) or o <= 0:
                continue
            cls = share_class_from_name(nm)
            rows.append({"security_id": f"{code}-{cls}-0000", "type": "bonus", "ref_date": d,
                         "factor": o / (o + n), "detail": f"{n:g} per {o:g}"})
    return pd.DataFrame(rows), unmatched


def load_rights(root: Path, m: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    rows, unmatched = [], []
    path = root / "04Rights Issues.xls"
    cols = ["company_name", "propotion"]
    wb = read_workbook(path, cols, required=["company_name", "propotion"], sheets=_year_sheets(path))
    for name, df in wb.items():
        y = re.search(r"(20\d\d)", str(name))
        if not y or y.group(1) not in YEARS:
            continue
        for _, r in df.iterrows():
            nm = r["company_name"]
            code = lookup_code(nm, m)
            prop = str(r.get("propotion", "")).strip()
            mm = re.match(r"^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$", prop)
            d = _date(r.get("allotment_date"))
            s = _num(r.get("consideration_per_share"))
            if code is None:
                unmatched.append(f"rights {y.group(1)}: {nm}")
                continue
            if not mm or pd.isna(d) or np.isnan(s):
                continue
            new, old = float(mm.group(1)), float(mm.group(2))
            cls = share_class_from_name(nm)
            rows.append({"security_id": f"{code}-{cls}-0000", "type": "rights", "ref_date": d,
                         "factor": np.nan, "new": new, "old": old, "subscription": s,
                         "detail": f"{new:g} for {old:g} at Rs {s:g}"})
    return pd.DataFrame(rows), unmatched


# --------------------------------------------------------------------------- dating and applying

def _locate_jump(px: pd.Series, ref: pd.Timestamp, target_logret: float, before: int, after: int,
                 tol: float) -> tuple[pd.Timestamp | None, float]:
    """Trading day in [ref-before, ref+after] whose close-to-close log return is closest to
    ``target_logret``; None if none is within ``tol``."""
    if px.empty:
        return None, np.nan
    dates = px.index
    pos = dates.searchsorted(ref)
    lo, hi = max(1, pos - before), min(len(dates) - 1, pos + after)
    best, best_err = None, np.inf
    lr = np.log(px.values[1:] / px.values[:-1])
    for i in range(lo, hi + 1):
        if i >= len(dates):
            break
        err = abs(lr[i - 1] - target_logret)
        if err < best_err:
            best, best_err = dates[i], err
    return (best, best_err) if best_err <= tol else (None, best_err)


@dataclass
class AdjustReport:
    events_by_type: dict
    dating_method: dict
    unmatched_names: list
    residual_jumps_gt_25pct: int


def build_events(root: Path, prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    m = name_map(root)
    splits = load_splits(root)
    divs = load_dividends(root)
    bonus, um1 = load_bonus(root, m)
    rights, um2 = load_rights(root, m)
    closes = {sid: g.set_index("date")["close"].sort_index() for sid, g in prices.groupby("security_id")}
    out = []
    for df in (splits, bonus, rights, divs):
        for r in df.to_dict("records"):
            px = closes.get(r["security_id"])
            if px is None or px.empty:
                continue
            if r["ref_date"] < px.index[0] - pd.Timedelta(days=40) or r["ref_date"] > px.index[-1] + pd.Timedelta(days=10):
                continue
            t = r["type"]
            if t == "dividend":
                r["ex_date"], r["dating"] = r["ref_date"], "file ex-date"
            elif t == "split":
                d, _ = _locate_jump(px, r["ref_date"], np.log(r["factor"]), 5, 5, 0.25)
                r["ex_date"], r["dating"] = (d, "price jump") if d is not None else (r["ref_date"], "file effective date")
            else:
                if t == "rights":
                    pos = max(0, px.index.searchsorted(r["ref_date"]) - 1)
                    p = float(px.iloc[pos])
                    r["factor"] = (r["old"] * p + r["new"] * r["subscription"]) / ((r["old"] + r["new"]) * p)
                if not (0 < r["factor"] < 1):
                    continue
                tgt = np.log(r["factor"])
                tol = max(0.02, 0.5 * abs(tgt))
                d, _ = _locate_jump(px, r["ref_date"], tgt, 20, 5, tol) if abs(tgt) >= 0.04 else (None, np.nan)
                r["ex_date"], r["dating"] = (d, "price jump") if d is not None else (r["ref_date"], "file allotment date")
            out.append(r)
    ev = pd.DataFrame(out)
    ev = ev[(ev["factor"] > 0) & (ev["factor"] < 1.0001)]
    return ev.sort_values(["security_id", "ex_date"]).reset_index(drop=True), um1 + um2


def apply_adjustment(prices: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add adj_open/high/low/close and adj_shares (split/bonus/rights only) to ``prices``."""
    p = prices.sort_values(["security_id", "date"]).copy()
    p["adj_factor"] = 1.0
    p["share_factor"] = 1.0
    for sid, ev in events.groupby("security_id"):
        mask = p["security_id"] == sid
        d = p.loc[mask, "date"]
        f = np.ones(mask.sum())
        sf = np.ones(mask.sum())
        for r in ev.itertuples():
            before = (d < r.ex_date).values
            f[before] *= r.factor
            if r.type in ("split", "bonus", "rights"):
                sf[before] *= r.factor
        p.loc[mask, "adj_factor"] = f
        p.loc[mask, "share_factor"] = sf
    for c in ("open", "high", "low", "close"):
        p[f"adj_{c}"] = p[c] * p["adj_factor"]
    p["adj_shares"] = p["shares"] / p["share_factor"]
    return p
