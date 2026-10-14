"""Load the dashboard tables (dist/db/*.parquet) into Neon PostgreSQL. Run on your laptop.

    python -m pip install pandas pyarrow sqlalchemy "psycopg[binary]"
    set ADMIN_DATABASE_URL=postgresql://<owner>:<password>@<host>/<db>?sslmode=require   (PowerShell: $env:ADMIN_DATABASE_URL="...")
    python load_to_postgres.py <folder with the .parquet files>

Idempotent and safe (NFR03): every table is first written to a staging name and swapped in inside
one transaction, so a failed load never leaves the published tables half-written. It then creates
a read-only role for the dashboard if READONLY_PASSWORD is set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

INDEXES = {
    "daily_price": ["ticker", "date"], "prediction": ["ticker", "as_of", "horizon"],
    "shap_contribution": ["ticker", "as_of", "horizon"], "var_estimate": ["ticker", "as_of"],
    "risk_parameters": ["ticker", "as_of"], "index_level": ["date"], "market_breadth": ["date"],
    "market_risk": ["date"], "sentiment_daily": ["ticker", "date"], "article": ["ticker", "trading_day"],
}


def main(folder: str):
    url = os.environ["ADMIN_DATABASE_URL"]
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    eng = sa.create_engine(url.replace("postgresql://", "postgresql+psycopg://", 1))
    files = sorted(Path(folder).glob("*.parquet"))
    with eng.begin() as con:
        for f in files:
            name = f.stem
            df = pd.read_parquet(f)
            df.to_sql(f"{name}__new", con, if_exists="replace", index=False, method="multi", chunksize=5000)
            if name in INDEXES:
                cols = ", ".join(INDEXES[name])
                con.execute(sa.text(f'CREATE INDEX ON "{name}__new" ({cols})'))
            con.execute(sa.text(f'DROP TABLE IF EXISTS "{name}"'))
            con.execute(sa.text(f'ALTER TABLE "{name}__new" RENAME TO "{name}"'))
            print(f"{name}: {len(df)} rows")
        pw = os.environ.get("READONLY_PASSWORD")
        if pw:
            exists = con.execute(sa.text("SELECT 1 FROM pg_roles WHERE rolname = 'dashboard_ro'")).scalar()
            if not exists:
                con.execute(sa.text(f"CREATE ROLE dashboard_ro LOGIN PASSWORD '{pw}'"))
            con.execute(sa.text("GRANT USAGE ON SCHEMA public TO dashboard_ro"))
            con.execute(sa.text("GRANT SELECT ON ALL TABLES IN SCHEMA public TO dashboard_ro"))
            con.execute(sa.text("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO dashboard_ro"))
            print("read-only role dashboard_ro ready")
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
