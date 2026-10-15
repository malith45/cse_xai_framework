"""Load the dashboard tables (*.parquet in this folder) into Neon PostgreSQL. Run on your laptop
(double-click RUN_LOAD.bat). It asks for the Neon connection string (input hidden, never stored),
loads every table, creates a read-only login "dashboard_ro" with a random password, and prints the
read-only connection string to paste into Streamlit secrets.

Idempotent and safe (NFR03): every table is first written to a staging name and swapped in inside
one transaction, so a failed load never leaves the published tables half-written.
"""

from __future__ import annotations

import getpass
import json
import time
import urllib.request
import os
import secrets
import string
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import sqlalchemy as sa

INDEXES = {
    "daily_price": ["ticker", "date"], "prediction": ["ticker", "as_of", "horizon"],
    "shap_contribution": ["ticker", "as_of", "horizon"], "var_estimate": ["ticker", "as_of"],
    "risk_parameters": ["ticker", "as_of"], "index_level": ["date"], "market_breadth": ["date"],
    "market_risk": ["date"], "sentiment_daily": ["ticker", "date"], "article": ["ticker", "trading_day"],
}


def _copy_insert(table, conn, keys, data_iter):
    """pandas to_sql method that streams rows with COPY (much faster than INSERT over the internet)."""
    cols = ", ".join(f'"{k}"' for k in keys)
    with conn.connection.cursor() as cur:
        with cur.copy(f'COPY "{table.name}" ({cols}) FROM STDIN') as cp:
            for row in data_iter:
                cp.write_row(row)


def _clean(raw: str) -> str:
    """Drop invisible characters a console paste can add (e.g. Ctrl+V captured as \\x16), spaces, line
    breaks, surrounding quotes, and a leading "psql" from the Neon console snippet."""
    url = "".join(ch for ch in (raw or "") if ch.isprintable() and not ch.isspace())
    return url.removeprefix("psql").strip("'\"")


def _from_clipboard() -> str:
    """Windows: read the connection string the user copied (Neon console -> Copy), then clear the clipboard."""
    if os.name != "nt":
        return ""
    import subprocess
    try:
        clip = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
                              capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return ""
    url = _clean(clip)
    if url.startswith(("postgres://", "postgresql://")):
        subprocess.run("echo off| clip", shell=True)   # do not leave the password on the clipboard
        return url
    return ""


def main(folder: str):
    url = _clean(os.environ.get("ADMIN_DATABASE_URL", "")) or _from_clipboard()
    if url:
        print(f"Using the connection string from your clipboard (database host {urlsplit(url).hostname}).")
    else:
        print("No connection string found on the clipboard. In Neon, click Copy next to the connection string,")
        print("then come back here. Or paste it once below (right-click) and press Enter.")
        url = _clean(getpass.getpass("Connection string (hidden): "))
    if not url.startswith(("postgres://", "postgresql://")):
        sys.exit("That does not look like a Postgres connection string (it should start with postgresql://).")
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    files = sorted(Path(folder).glob("*.parquet"))
    if not files:
        sys.exit(f"No .parquet files found in {folder}")
    eng = sa.create_engine(url.replace("postgresql://", "postgresql+psycopg://", 1))
    for attempt in range(3):          # a sleeping Neon compute can drop the first connection
        try:
            with eng.connect() as c:
                c.execute(sa.text("SELECT 1"))
            break
        except Exception as e:  # noqa: BLE001
            print(f"Direct connection (port 5432) failed, attempt {attempt + 1}/3: {str(e).splitlines()[0][:120]}")
            time.sleep(5)
    else:
        print("Your network blocks port 5432. Switching to Neon's HTTPS endpoint (port 443)...")
        return load_over_https(url, files)
    print(f"Loading {len(files)} tables (a few minutes)...")
    with eng.begin() as con:
        for f in files:
            name = f.stem
            df = pd.read_parquet(f)
            df.to_sql(f"{name}__new", con, if_exists="replace", index=False, method=_copy_insert, chunksize=50000)
            if name in INDEXES:
                cols = ", ".join(INDEXES[name])
                con.execute(sa.text(f'CREATE INDEX ON "{name}__new" ({cols})'))
            con.execute(sa.text(f'DROP TABLE IF EXISTS "{name}"'))
            con.execute(sa.text(f'ALTER TABLE "{name}__new" RENAME TO "{name}"'))
            print(f"{name}: {len(df)} rows")
        alphabet = string.ascii_letters + string.digits
        pw = "".join(secrets.choice(alphabet) for _ in range(32))
        exists = con.execute(sa.text("SELECT 1 FROM pg_roles WHERE rolname = 'dashboard_ro'")).scalar()
        reset = os.environ.get("RESET_READONLY_PASSWORD") == "1"
        if exists and not reset:
            pw = None   # keep the existing password: the Streamlit secret stays valid
        elif exists:
            con.execute(sa.text(f"ALTER ROLE dashboard_ro WITH LOGIN PASSWORD '{pw}'"))
        else:
            con.execute(sa.text(f"CREATE ROLE dashboard_ro LOGIN PASSWORD '{pw}'"))
        db = con.execute(sa.text("SELECT current_database()")).scalar()
        con.execute(sa.text(f'GRANT CONNECT ON DATABASE "{db}" TO dashboard_ro'))
        con.execute(sa.text("GRANT USAGE ON SCHEMA public TO dashboard_ro"))
        con.execute(sa.text("GRANT SELECT ON ALL TABLES IN SCHEMA public TO dashboard_ro"))
        con.execute(sa.text("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO dashboard_ro"))
    if pw is None:
        print("\nAll tables loaded. The read-only login dashboard_ro already existed, so its password is unchanged")
        print("and your Streamlit secret still works. (To issue a new one: set RESET_READONLY_PASSWORD=1 and re-run.)")
        return
    parts = urlsplit(url)
    host = parts.hostname + (f":{parts.port}" if parts.port else "")
    ro = urlunsplit((parts.scheme.replace("+psycopg", ""), f"dashboard_ro:{pw}@{host}", parts.path, parts.query, ""))
    print("\nAll tables loaded. Read-only connection string for Streamlit secrets (DATABASE_URL):\n")
    print(ro)
    print("\nCopy it now; it is not saved anywhere. Re-running this loader creates a new one.")



# --------------------------------------------------------------------------- HTTPS fallback (port 443)

def _sql_type(dt) -> str:
    if pd.api.types.is_bool_dtype(dt):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(dt):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dt):
        return "DOUBLE PRECISION"
    if pd.api.types.is_datetime64_any_dtype(dt):
        return "TIMESTAMP"
    return "TEXT"


def _http(url: str, body: dict) -> dict:
    """Neon serverless HTTP endpoint: POST https://<host>/sql with the connection string in a header."""
    host = urlsplit(url).hostname
    endpoint = os.environ.get("NEON_HTTP_ENDPOINT") or f"https://{host}/sql"   # override only for tests
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Neon-Connection-String": url,
                                          "Neon-Raw-Text-Output": "true", "Neon-Array-Mode": "true"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code < 500 or attempt == 3:
                sys.exit(f"Neon HTTPS error {e.code}: {msg}")
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                sys.exit(f"Neon HTTPS connection failed: {e}")
        time.sleep(5 * (attempt + 1))


def _q(url, query, params=None):
    return _http(url, {"query": query, "params": params or []})


def _cell(v):
    if v is None or (isinstance(v, float) and v != v) or v is pd.NaT:
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, (bool,)):
        return "true" if v else "false"
    return str(v)


def load_over_https(url: str, files):
    _q(url, "SELECT 1")
    print(f"Connected over HTTPS. Loading {len(files)} tables (several minutes)...")
    for f in files:
        name = f.stem
        df = pd.read_parquet(f)
        new = f"{name}__new"
        cols = ", ".join(f'"{c}" {_sql_type(df[c].dtype)}' for c in df.columns)
        _q(url, f'DROP TABLE IF EXISTS "{new}"')
        _q(url, f'CREATE TABLE "{new}" ({cols})')
        ncol = len(df.columns)
        step = max(1, 30000 // ncol)
        names = ", ".join(f'"{c}"' for c in df.columns)
        for i in range(0, len(df), step):
            part = df.iloc[i:i + step]
            ph, params = [], []
            for row in part.itertuples(index=False):
                base = len(params)
                ph.append("(" + ", ".join(f"${base + k + 1}" for k in range(ncol)) + ")")
                params.extend(_cell(v) for v in row)
            _q(url, f'INSERT INTO "{new}" ({names}) VALUES ' + ", ".join(ph), params)
        if name in INDEXES:
            _q(url, f'CREATE INDEX ON "{new}" ({", ".join(INDEXES[name])})')
        _http(url, {"queries": [{"query": f'DROP TABLE IF EXISTS "{name}"', "params": []},
                                {"query": f'ALTER TABLE "{new}" RENAME TO "{name}"', "params": []}]})
        print(f"{name}: {len(df)} rows")
    _finish_role_https(url)


def _finish_role_https(url: str):
    exists = _q(url, "SELECT 1 FROM pg_roles WHERE rolname = 'dashboard_ro'").get("rows")
    reset = os.environ.get("RESET_READONLY_PASSWORD") == "1"
    pw = None
    if not exists or reset:
        pw = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
        verb = "ALTER" if exists else "CREATE"
        _q(url, f"{verb} ROLE dashboard_ro WITH LOGIN PASSWORD '{pw}'")
    db = urlsplit(url).path.lstrip("/") or "neondb"
    for stmt in (f'GRANT CONNECT ON DATABASE "{db}" TO dashboard_ro', "GRANT USAGE ON SCHEMA public TO dashboard_ro",
                 "GRANT SELECT ON ALL TABLES IN SCHEMA public TO dashboard_ro",
                 "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO dashboard_ro"):
        _q(url, stmt)
    _print_ro(url, pw)


def _print_ro(url: str, pw):
    if pw is None:
        print("\nAll tables loaded. The read-only login dashboard_ro already existed, so its password is unchanged")
        print("and your Streamlit secret still works. (To issue a new one: set RESET_READONLY_PASSWORD=1 and re-run.)")
        return
    parts = urlsplit(url)
    host = parts.hostname + (f":{parts.port}" if parts.port else "")
    ro = urlunsplit((parts.scheme.replace("+psycopg", ""), f"dashboard_ro:{pw}@{host}", parts.path, parts.query, ""))
    print("\nAll tables loaded. Read-only connection string for Streamlit secrets (DATABASE_URL):\n")
    print(ro)
    print("\nCopy it now; it is not saved anywhere.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent))
