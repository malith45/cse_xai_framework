"""Trilingual news collector for the CSE dissertation (M3 input; FR04). Runs on your laptop.

Sources
  * Ada Derana Biz English (bizenglish.adaderana.lk) - English business news via its public
    WordPress REST API (title, date-time, full text in one call per 100 posts).
  * Hiru News sitemaps (www.hirunews.lk/sitemaps/) - English, Sinhala and Tamil editions. Each
    sitemap entry already carries the URL, headline and publication date, so every headline in the
    period is collected; the article page is then fetched only for market-relevant headlines.

Rules followed: robots.txt is checked before every host is used; one request per DELAY seconds per
host; a descriptive User-Agent; nothing is redistributed (full text stays on this computer - the
dashboard shows only headline, source, date and link, Chapter 4 §4.7 step 14).

Usage (PowerShell, from this folder):
    python -m pip install requests beautifulsoup4 lxml
    python news_scraper.py --start 2020-01-01 --end 2025-12-31 --out news_raw
Re-running resumes: finished pages and articles are skipped.
Outputs in --out: adaderana_biz_en.csv, hiru_<lang>_headlines.csv, hiru_<lang>_articles.csv, log.txt
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.robotparser
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

UA = "CSE-Dissertation-NewsCollector/1.0 (academic research, NIBM/Coventry University; non-commercial)"
DELAY = 1.0  # seconds between requests to the same host (Hiru robots.txt sets no crawl-delay)
TIMEOUT = 30

# Market relevance filter for Hiru headlines. Matched against the English URL slug (Hiru writes an
# English slug for all three editions) and against the headline itself.
SLUG_KEYWORDS = [
    "stock", "share", "shares", "cse", "aspi", "s-p-sl20", "sl20", "colombo-stock", "bourse", "market",
    "investor", "dividend", "profit", "loss", "earnings", "revenue", "economy", "economic", "gdp",
    "inflation", "rupee", "dollar", "exchange-rate", "interest-rate", "policy-rate", "central-bank",
    "cbsl", "imf", "debt", "bond", "treasury", "t-bill", "tax", "vat", "budget", "export", "import",
    "tourism", "remittance", "fuel-price", "electricity-tariff", "bank", "banks", "finance", "leasing",
    "insurance", "plc", "company", "companies", "business", "industry", "trade", "price", "prices",
    # companies frequently in the S&P SL20 (tickers and names as they appear in slugs)
    "john-keells", "keells", "jkh", "hayleys", "dialog", "commercial-bank", "combank", "sampath",
    "hatton-national", "hnb", "lolc", "browns", "melstacorp", "distilleries", "ceylon-tobacco", "ctc",
    "lanka-ioc", "ioc", "chevron", "teejay", "hemas", "sunshine", "cargills", "ndb", "national-development-bank",
    "dfcc", "access-engineering", "royal-ceramics", "dipped-products", "vallibel", "expolanka", "softlogic",
    "cic", "tokyo-cement", "aitken-spence", "lion-brewery", "ceylon-cold-stores", "nestle", "slt", "telecom",
    "people-s-leasing", "central-finance", "lb-finance", "hayleys-fabric", "watawala", "lanka-walltiles",
]
TITLE_KEYWORDS = [
    # English
    "stock", "share", "CSE", "ASPI", "S&P", "market", "dividend", "profit", "rupee", "inflation",
    "interest rate", "Central Bank", "IMF", "economy", "PLC",
    # Sinhala
    "කොටස්", "කොටස් වෙළෙඳපොළ", "කොළඹ කොටස්", "ආර්ථික", "ආර්ථිකය", "රුපියල", "ඩොලර්", "උද්ධමන",
    "පොලී", "මහ බැංකුව", "බැංකු", "ලාභ", "සමාගම", "ජාත්‍යන්තර මූල්‍ය අරමුදල", "අයවැය", "බදු",
    # Tamil
    "பங்கு", "பங்குச் சந்தை", "கொழும்பு பங்கு", "பொருளாதார", "ரூபா", "டொலர்", "பணவீக்க", "வட்டி",
    "மத்திய வங்கி", "வங்கி", "இலாப", "நிறுவன", "சர்வதேச நாணய நிதிய", "வரவு செலவு", "வரி",
]


class Polite:
    """requests.Session with robots.txt checks and a per-host delay."""

    def __init__(self, log):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.last: dict[str, float] = {}
        self.log = log

    def allowed(self, url: str) -> bool:
        host = urlparse(url).scheme + "://" + urlparse(url).netloc
        if host not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.s.get(host + "/robots.txt", timeout=TIMEOUT)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except requests.RequestException:
                rp.parse([])
            self.robots[host] = rp
        return self.robots[host].can_fetch(UA, url)

    def get(self, url: str, **kw) -> requests.Response | None:
        if not self.allowed(url):
            self.log(f"robots.txt disallows {url}")
            return None
        host = urlparse(url).netloc
        wait = DELAY - (time.time() - self.last.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            try:
                r = self.s.get(url, timeout=TIMEOUT, **kw)
                self.last[host] = time.time()
                if r.status_code == 200:
                    return r
                if r.status_code in (404, 410):
                    return None
                self.log(f"HTTP {r.status_code} {url}")
            except requests.RequestException as e:
                self.log(f"error {e} {url}")
            time.sleep(5 * (attempt + 1))
        return None


def clean_html(s: str) -> str:
    return " ".join(BeautifulSoup(html.unescape(s or ""), "lxml").get_text(" ").split())


# --------------------------------------------------------------------------- Ada Derana Biz English

def adaderana(http: Polite, start: str, end: str, out: Path, log):
    path = out / "adaderana_biz_en.csv"
    seen = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            seen = {r["url"] for r in csv.DictReader(f)}
    new = not path.exists()
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, ["source", "language_hint", "url", "published_at", "title", "body", "category", "fetched_at"])
    if new:
        w.writeheader()
    page = 1
    api = "https://bizenglish.adaderana.lk/wp-json/wp/v2/posts"
    while True:
        r = http.get(api, params={"after": f"{start}T00:00:00", "before": f"{end}T23:59:59", "per_page": 100,
                                  "page": page, "orderby": "date", "order": "asc",
                                  "_fields": "id,date,link,title,content,categories"})
        if r is None:
            break
        try:
            posts = r.json()
        except json.JSONDecodeError:
            break
        if not posts:
            break
        for p in posts:
            if p["link"] in seen:
                continue
            w.writerow({"source": "Ada Derana Biz English", "language_hint": "en", "url": p["link"],
                        "published_at": p["date"], "title": clean_html(p["title"]["rendered"]),
                        "body": clean_html(p["content"]["rendered"])[:5000],
                        "category": ",".join(map(str, p.get("categories", []))),
                        "fetched_at": datetime.now().isoformat(timespec="seconds")})
        f.flush()
        total_pages = int(r.headers.get("X-WP-TotalPages", page))
        log(f"Ada Derana page {page}/{total_pages}")
        if page >= total_pages:
            break
        page += 1
    f.close()


# --------------------------------------------------------------------------- Hiru News

ENTRY = re.compile(r"<url>(.*?)</url>", re.S)
LOC = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.S)
TITLE = re.compile(r"<(?:news:)?title>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</(?:news:)?title>", re.S)
PDATE = re.compile(r"<(?:news:publication_date|lastmod)>\s*(.*?)\s*</", re.S)


def parse_sitemap(text: str) -> list[dict]:
    rows = []
    for e in ENTRY.findall(text):
        loc = LOC.search(e)
        if not loc:
            continue
        t = TITLE.search(e)
        d = PDATE.search(e)
        rows.append({"url": html.unescape(loc.group(1)), "title": html.unescape(t.group(1)).strip() if t else "",
                     "published_at": d.group(1) if d else ""})
    return rows


def sitemap_chunks(http: Polite, lang: str) -> list[str]:
    r = http.get("https://www.hirunews.lk/sitemaps/")
    if r is None:
        r = http.get("https://www.hirunews.lk/sitemap.xml")
    urls = re.findall(r"<loc>\s*(.*?)\s*</loc>", r.text if r else "")
    chunks = [u for u in urls if re.search(rf"/{lang}-\d+\.xml$", u)]
    return sorted(chunks, key=lambda u: int(re.search(r"-(\d+)\.xml$", u).group(1)))


def chunk_dates(http: Polite, url: str) -> tuple[str, str, list[dict]]:
    r = http.get(url)
    rows = parse_sitemap(r.text) if r else []
    ds = sorted(x["published_at"][:10] for x in rows if x["published_at"])
    return (ds[0], ds[-1], rows) if ds else ("", "", rows)


def relevant(url: str, title: str) -> bool:
    slug = urlparse(url).path.lower()
    if "/business/" in slug:
        return True
    if any(re.search(rf"(^|[/-]){re.escape(k)}($|[/-])", slug) for k in SLUG_KEYWORDS):
        return True
    return any(k.lower() in title.lower() for k in TITLE_KEYWORDS)


def hiru(http: Polite, lang: str, start: str, end: str, out: Path, log, fetch_articles: bool):
    chunks = sitemap_chunks(http, lang)
    log(f"Hiru {lang}: {len(chunks)} sitemap chunks")
    if not chunks:
        return
    # binary search for the first chunk whose newest entry is on/after `start`
    lo, hi = 0, len(chunks) - 1
    cache = {}
    while lo < hi:
        mid = (lo + hi) // 2
        cache[mid] = chunk_dates(http, chunks[mid])
        if cache[mid][1] and cache[mid][1] < start:
            lo = mid + 1
        else:
            hi = mid
    hpath = out / f"hiru_{lang}_headlines.csv"
    done_chunks = set()
    state = out / f"hiru_{lang}_state.json"

    def key(chunk: str) -> str:
        # a chunk is "done" only for the date range it was read with (a chunk that straddles two
        # ranges, e.g. Dec 2022 / Jan 2023, must be read again for the other range)
        return f"{chunk}|{start}|{end}"

    if state.exists():
        # entries written before ranges were recorded came from the 2023-01-01..2025-12-31 run
        done_chunks = {x if "|" in x else f"{x}|2023-01-01|2025-12-31" for x in json.load(open(state))}
    newfile = not hpath.exists()
    hf = open(hpath, "a", newline="", encoding="utf-8")
    hw = csv.DictWriter(hf, ["source", "language_hint", "url", "published_at", "title", "relevant"])
    if newfile:
        hw.writeheader()
    for i in range(lo, len(chunks)):
        if key(chunks[i]) in done_chunks:
            continue
        first, last, rows = cache.get(i) or chunk_dates(http, chunks[i])
        if first and first > end:
            break
        for x in rows:
            d = x["published_at"][:10]
            if d and start <= d <= end:
                hw.writerow({"source": "Hiru News", "language_hint": lang, "url": x["url"],
                             "published_at": x["published_at"], "title": x["title"],
                             "relevant": int(relevant(x["url"], x["title"]))})
        hf.flush()
        done_chunks.add(key(chunks[i]))
        json.dump(sorted(done_chunks), open(state, "w"))
        log(f"Hiru {lang}: chunk {i + 1}/{len(chunks)} {first}..{last}")
    hf.close()
    if fetch_articles:
        hiru_articles(http, lang, out, log)


def extract_body(soup: BeautifulSoup) -> str:
    """Hiru article text lives in <div id="this-article" class="description-content"> as plain text with
    <br> line breaks (verified on 2023, 2024 and 2026 pages); fall back to long <p> paragraphs."""
    node = soup.select_one("#this-article") or soup.select_one(".description-content")
    if node is not None:
        text = " ".join(node.get_text(" ", strip=True).split())
        if len(text) > 40:
            return text[:5000]
    for t in soup(["script", "style", "nav", "header", "footer", "aside"]):
        t.decompose()
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return " ".join(p for p in paras if len(p) > 40 and "All rights Reserved" not in p)[:5000]


def refetch_empty(http: Polite, lang: str, out: Path, log):
    """Re-fetch Hiru articles saved without body text (first run used a parser that missed the
    #this-article container). Rewrites hiru_<lang>_articles.csv; safe to stop and re-run."""
    apath = out / f"hiru_{lang}_articles.csv"
    done_path = out / f"hiru_{lang}_refetched.csv"
    if not apath.exists():
        return
    with open(apath, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys()) if rows else []
    done = {}
    if done_path.exists():
        with open(done_path, encoding="utf-8") as f:
            done = {r["url"]: r["body"] for r in csv.DictReader(f)}
    todo = [r for r in rows if len((r.get("body") or "").strip()) < 40 and r["url"] not in done]
    log(f"Hiru {lang}: {len(todo)} articles to re-fetch for body text")
    newfile = not done_path.exists()
    df = open(done_path, "a", newline="", encoding="utf-8")
    dw = csv.DictWriter(df, ["url", "body"])
    if newfile:
        dw.writeheader()
    for n, r in enumerate(todo, 1):
        resp = http.get(r["url"])
        body = extract_body(BeautifulSoup(resp.text, "lxml")) if resp is not None else ""
        dw.writerow({"url": r["url"], "body": body})
        done[r["url"]] = body
        if n % 25 == 0:
            df.flush()
            log(f"Hiru {lang}: re-fetched {n}/{len(todo)}")
    df.close()
    for r in rows:
        if len((r.get("body") or "").strip()) < 40 and done.get(r["url"]):
            r["body"] = done[r["url"]]
    tmp = apath.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(apath)
    filled = sum(len((r.get("body") or "").strip()) >= 40 for r in rows)
    log(f"Hiru {lang}: {filled}/{len(rows)} articles now have body text")


def hiru_articles(http: Polite, lang: str, out: Path, log):
    hpath, apath = out / f"hiru_{lang}_headlines.csv", out / f"hiru_{lang}_articles.csv"
    with open(hpath, encoding="utf-8") as f:
        todo = [r for r in csv.DictReader(f) if r["relevant"] == "1"]
    seen = set()
    if apath.exists():
        with open(apath, encoding="utf-8") as f:
            seen = {r["url"] for r in csv.DictReader(f)}
    newfile = not apath.exists()
    af = open(apath, "a", newline="", encoding="utf-8")
    aw = csv.DictWriter(af, ["source", "language_hint", "url", "published_at", "page_datetime", "title", "body", "fetched_at"])
    if newfile:
        aw.writeheader()
    todo = [r for r in todo if r["url"] not in seen]
    log(f"Hiru {lang}: {len(todo)} relevant articles to fetch")
    for n, r in enumerate(todo, 1):
        resp = http.get(r["url"])
        if resp is None:
            continue
        soup = BeautifulSoup(resp.text, "lxml")
        body = extract_body(soup)
        meta_time = soup.find("meta", attrs={"property": "article:published_time"})
        page_dt = meta_time["content"] if meta_time and meta_time.get("content") else ""
        if not page_dt:
            m = re.search(r"(\d{1,2}\s+\w+\s+20\d\d)\s*(?:-|,)?\s*(\d{1,2}:\d{2}\s*(?:[AaPp][Mm])?)?", soup.get_text(" "))
            page_dt = " ".join(x for x in (m.groups() if m else ()) if x)
        aw.writerow({"source": "Hiru News", "language_hint": lang, "url": r["url"], "published_at": r["published_at"],
                     "page_datetime": page_dt, "title": r["title"], "body": body,
                     "fetched_at": datetime.now().isoformat(timespec="seconds")})
        if n % 25 == 0:
            af.flush()
            log(f"Hiru {lang}: fetched {n}/{len(todo)}")
    af.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", default="news_raw")
    ap.add_argument("--only", nargs="*", default=["adaderana", "hiru-en", "hiru-si", "hiru-ta"])
    ap.add_argument("--headlines-only", action="store_true", help="skip fetching article pages")
    ap.add_argument("--refetch-empty", action="store_true", help="only re-fetch Hiru articles saved without body text")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    http = Polite(log)
    if a.refetch_empty:
        for lang in ("english", "sinhala", "tamil"):
            try:
                refetch_empty(http, lang, out, log)
            except KeyboardInterrupt:
                log("stopped by user - re-run the same command to resume")
                sys.exit(1)
        log("finished")
        return
    jobs = {"adaderana": lambda: adaderana(http, a.start, a.end, out, log),
            "hiru-en": lambda: hiru(http, "english", a.start, a.end, out, log, not a.headlines_only),
            "hiru-si": lambda: hiru(http, "sinhala", a.start, a.end, out, log, not a.headlines_only),
            "hiru-ta": lambda: hiru(http, "tamil", a.start, a.end, out, log, not a.headlines_only)}
    for name in a.only:
        try:
            jobs[name]()
        except KeyboardInterrupt:
            log("stopped by user - re-run the same command to resume")
            sys.exit(1)
        except Exception as e:  # noqa: BLE001 - keep going with the other sources
            log(f"{name} failed: {e!r}")
    log("finished")


if __name__ == "__main__":
    main()
