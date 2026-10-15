"""Load the dashboard tables (*.parquet in this folder) into Neon PostgreSQL. Run on your laptop
(double-click RUN_LOAD.bat). It asks for the Neon connection string (input hidden, never stored),
loads every table, creates a read-only login "dashboard_ro" with a random password, and prints the
read-only connection string to paste into Streamlit secrets.

Idempotent and safe (NFR03): every table is first written to a staging name and swapped in inside
one transaction, so a failed load never leaves the published tables half-written.
"""

from __future__ import annotations

import getpass
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
    eng = sa.create_engine(url.replace("postgresql://", "postgresql+psycopg://", 1))
    files = sorted(Path(folder).glob("*.parquet"))
    if not files:
        sys.exit(f"No .parquet files found in {folder}")
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


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent))
