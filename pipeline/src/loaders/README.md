# `loaders` — file-specific loaders for the CSE Data Library

Each loader uses `readers` for header detection, format dispatch and sheet ordering, and
`identifiers` for canonical security IDs, and adds only the semantics of its own file.

## `public_float.py` — Public Holding (free float), file 17

**Source:** `17 Public Holding-Quarterly/Public Holding/Public Holding YYYY.xlsx`, 14 files 2012–2025,
four quarterly sheets each (55 sheets with data plus one Covid "No records" sheet).

**Output:** `load_public_float(folder)` → one tidy DataFrame, sorted by security then time:

| column | meaning |
|---|---|
| `security_id` | canonical `AAF-N-0000` via `identifiers.parse` (share classes never collapsed) |
| `company_name` | as printed (whitespace collapsed) |
| `year`, `quarter` | from the QUARTER cell, cross-checked against the sheet name |
| `quarter_end` | last calendar day of the quarter |
| `float_pct` | public float in **percentage points, 0–100 inclusive** |

`load_public_float(folder, with_reports=True)` also returns a `SheetReport` per sheet with the
detected convention, row count and any note found in the data area.

### Unit conventions and the detection rule

The float is stored two ways in the library:

| Convention | Example cells | Meaning |
|---|---|---|
| decimal fraction (numeric) | `0.1024`, `1`, `0` | 10.24 %, 100 %, 0 % |
| percentage string | `"2.84%"`, `"100%"` | 2.84 %, 100 % |

**The changeover is not by file and it is not monotonic.** Verified per sheet on 2026-09-19:

| Sheets | Convention |
|---|---|
| 2012 Q1 → 2016 Q3 | fraction |
| 2016 Q4 → 2018 Q2 | percent string |
| **2018 Q3** | fraction (reverts for one quarter) |
| 2018 Q4 → 2019 Q4 | percent string |
| 2020 Q1 | empty: "No records for the period … Due to the Covid…" note under the header |
| **2020 Q2** | **mixed**: 220 numeric fractions and 121 percent strings in one column |
| **2020 Q3** | fraction (reverts) |
| 2020 Q4 → 2022 Q2 | percent string |
| **2022 Q3** | fraction (reverts) |
| 2022 Q4 → 2025 Q4 | percent string |

Detection is therefore done **per sheet and per value**, from the values only, never from the year:

1. A cell matching `^number%$` is a percentage and is used as-is.
2. For the sheet's numeric cells, `detect_numeric_convention`: if **every** numeric value is ≤ 1.0 the
   sheet stores fractions and each is multiplied by 100; if **any** numeric value exceeds 1.0 the sheet
   stores percentage points (not observed in the library, but supported).
3. Anything else (blank, NaN, text without `%`) is an error.
4. **Range assertion:** every normalised value must satisfy 0 ≤ v ≤ 100. Violation raises
   `FloatOutOfRange` naming file, sheet, row index and security. Nothing is clamped, skipped or warned.
5. **Plausibility guard:** the sheet's median must lie in 5–60 %. Observed medians are 21.7–26.6 %.
   A misdetected convention would give about 0.25 % (fractions read as points) or about 2,500 %
   (points read as fractions), so the guard catches the one ambiguous case that rule 2 cannot: a
   points sheet in which every value happens to be ≤ 1.

The mixed 2020 Q2 sheet resolves correctly under these rules: its numeric values (max 0.9986, plus
integer `0` and `1`) are fractions and its strings are percentages. The result matches 2019 Q4 and
2020 Q3 for the same companies (MFL `35.2%` / `0.352` / `0.352`; BLUE.X `100%` / `1` / `1.0`).

### Other quirks handled

- **QUARTER cell** is `2012-4`, `2018-1`, `2016 - 1` (spaces) or `2025-01` (zero-padded). The second
  part is always the **quarter number**, never a month. `parse_quarter_code` handles all of them and the
  result must agree with the sheet name, otherwise `QuarterParseError`.
- **Sheet names** `2012 Q4`, `2018_Q1`, `1-2025` and the 2012–2017 reverse ordering are handled by
  `readers.read_workbook`, which returns quarters oldest-first.
- **Column spellings** `COMPANY NAME`/`COMPANY_NAME`, `SECURITY ID`/`SECURITY` and `PUBLIC FLOAT`/
  `PUBLIC_FLOAT` are unified by `readers.normalise_column_name`.
- **IDs** such as `VLL.x0000` (lower-case class) become `VLL-X-0000`. An ID without a share class, or a
  malformed one, is an error rather than a guess.
- **Duplicate rows.** From 2024 Q3 to 2025 Q4 every sheet lists VIDULLANKA's non-voting share twice:
  one row frozen at `70.29%` beside a live row that moves 70.5 → 72.85 %. Both spellings
  (`VLL.x0000`, `VLL.X0000`) canonicalise to `VLL-X-0000`. Casing is a typo, not a business rule, so
  it is never used to choose. Resolution happens **after the whole series is loaded**:
  - `duplicate_policy="error"` (default): halt and print every candidate row with file, sheet, row,
    raw ID and raw value.
  - `duplicate_policy="prefer_varying_value"`: for each security with duplicates, the value that is
    identical in **every** duplicated quarter is the stale copy and is dropped; the row whose value
    varies across quarters is kept. It needs at least two duplicated quarters. If no candidate is
    frozen (both vary), more than one is (both frozen), or a quarter does not reduce to exactly one
    row, it halts and prints all rows. Every drop is recorded in `SheetReport.notes`.
  - `audit_duplicates(folder)` lists every candidate row side by side, with nothing dropped, for
    manual verification.
  **Decision (author, 2026-09-19): use `prefer_varying_value`.**
- The file named `Public Holding 2023 (2).xlsx` is picked up by the glob like any other.

### Failure policy

Every error names the file and sheet, and the row where applicable. The loader halts on the first
problem and never returns partially normalised data.

```bash
.venv/Scripts/python.exe -m pytest tests/test_public_float.py -q
```
