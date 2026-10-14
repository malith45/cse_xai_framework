# `readers` — spreadsheet readers for the CSE Data Library

Part of **M1 Data Ingestion & Validation** (Chapter 4 §4.7 step 1). Turns any human-formatted
CSE workbook into a DataFrame with canonical column names, or fails loudly.

## What the files do to you (all verified on the real 2025 Q4 library)

| Problem | Example | Handling |
|---|---|---|
| Title text and blank rows above the header, at varying depths | Public Holding 2012: header on row 4; 2018: row 3; 2015/2021–2025: row 2; price lists: row 3, 4 or 5 | `find_header_row` scans up to 40 rows for the first row containing every required column name. Never a fixed index. |
| Three file formats | 2021 `.xls`, 2023 `.csv`, 2025 `.xlsx` in the same price folder | `load_raw_sheets` dispatches on extension: xlrd / openpyxl / pandas CSV with encoding fallback |
| Sheet-name conventions change | `2012 Q4`, `2018_Q1`, `2012Q2`, `1-2025`, `2025`, `'2018 '`, `Daily Share Price Q1` | `parse_sheet_name` → `SheetPeriod(year, quarter)`; `1-2025` is **quarter 1**, not January |
| Reverse chronological sheet order | Dividends, Bonus, Public Holding 2012–2017 list newest first | `read_workbook(chronological=True)` returns oldest-first |
| "No records" placeholder sheets | Bonus Issues 2024 and 2025 | `is_empty_sheet` → empty DataFrame with the expected columns, never a crash |
| Column spelling variants | `COMPANY NAME` / `COMPANY_NAME` / `NAME`; `SECURITY` / `SECURITY ID`; `PUBLIC FLOAT` / `PUBLIC_FLOAT`; `EX PRICE` / `EX_PRICE`; `PRICE HIGH (Rs.)` | `normalise_column_name`: strip, collapse whitespace, drop unit suffixes, snake_case, alias map |
| Leading/trailing spaces in header cells and data | `"  MAIN TYPE  "`, company ID `" "` | headers normalised; string cells stripped; all-blank rows dropped (removes the 2025 separator rows) |
| Trailing empty columns | 2023 CSV has four | unnamed all-empty columns dropped |
| **Ragged rows from unquoted thousands separators** | 2023 CSV: every price ≥ Rs 1,000 is written `1,225.00` unquoted, so 1,137 rows (CTEA, CINS N and X, NEH, SLND, SOY, BREW, HARI, NEST, PHAR, **CTC**, GOOD, SWAD, SHAL, INDO) have 13–16 fields and shift right | spilled cells are kept in `unnamed_12…15`; `spilled_rows(df)` returns the affected index. The reader never repairs or drops them; that decision belongs to the price parser and the author |
| Duplicate header names in one row | Quoted Securities 2014 has two `SECURITY ID` tables side by side | second becomes `security_id_2` (deterministic, never silently merged) |

## Canonical column names

Lower snake_case. Units in parentheses are removed. Known variants are mapped by `COLUMN_ALIASES`
in `columns.py`; the most important:

| Canonical | Variants seen |
|---|---|
| `company_code` | `COMPANY ID`, `COMPANY CODE` |
| `company_name` | `COMPANY NAME`, `COMPANY_NAME`, `NAME`, `Company` |
| `security_id` | `SECURITY`, `SECURITY ID`, `SECURITY TYPE` (Market Cap file holds full IDs there) |
| `public_float` | `PUBLIC FLOAT`, `PUBLIC_FLOAT` |
| `price_high`, `price_low`, `close_price`, `open_price`, `trade_volume`, `share_volume`, `turnover` | same with `(Rs.)` / `(No.)` suffixes |
| `ex_price` | `EX PRICE`, `EX_PRICE` |
| `date_of_announcement` | `DATE OF ANNOUNCEMENT`, `Date of Initial Announcement` |
| `allotment_date` | `Allotment Date`, `Date of P/Allotment` |

Values are **not** type-converted here. Excel cells keep native types (datetimes, floats); CSV cells
are all strings (e.g. `"10,261"`). Dates that are text in one year and datetimes in another, and
`SUB TYPE` as `"0000"` vs `0.0`, are the downstream parser's job (see `identifiers` for IDs).

## API

```python
from readers import read_sheet, read_workbook, parse_sheet_name, HeaderNotFound

PH = ["quarter", "company_name", "security_id", "public_float"]
df = read_sheet(path, "2012 Q4", PH)                # header found on row 4
frames = read_workbook(path, PH)                    # {"2012 Q1": df, "2012 Q2": df, ...} oldest first

bad = spilled_rows(df)                              # rows whose cells overflowed the header (2023 CSV)

parse_sheet_name("1-2025")     # SheetPeriod(year=2025, quarter=1)
parse_sheet_name("2018_Q1")    # SheetPeriod(year=2018, quarter=1)
parse_sheet_name("HighLow")    # None
```

`required=` lets you accept a subset when a column exists only in some years (e.g. Dividends 2010 has
`Rate & Remarks` instead of `RATE OF DIVIDEND` + `REMARKS`).

## Failure policy

If no row within `max_depth` contains all required columns **and** the sheet is not a placeholder,
`HeaderNotFound` is raised. Its message names the file and sheet, lists the required columns, and
prints the normalised contents of the first rows so the mismatch is obvious. Wrong columns are never
returned silently.

A sheet counts as a placeholder when it contains a cell matching "No record(s)" or has at most five
non-blank cells in total.

## Not handled here

- Sheets that contain **several sub-tables** separated by title rows (New Listings & De-listings,
  Rights Issues with a totals row, Beta with two side-by-side tables). `read_sheet` returns the first
  table's header; splitting sub-tables is a file-specific concern.
- The 2001–2019 per-company **block layout** of the daily price lists (out of the study period).
- Wide-format monthly files (dates across columns).
- Type coercion, ID normalisation (`identifiers`), and name→code matching.

## Tests

`tests/test_readers.py` runs against the real files under `CSE Docs`. They are skipped when that
folder is absent (it is licensed and never committed).

```bash
.venv/Scripts/python.exe -m pytest tests/test_readers.py -q
```
