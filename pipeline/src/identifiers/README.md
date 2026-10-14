# `identifiers` — CSE security identifier normaliser

Part of **M1 Data Ingestion & Validation** (Chapter 4 §4.7, step 1 "cleaning of malformed
records"). It gives every other module one key for a security regardless of which CSE Data Library
file the row came from.

## The problem

The same security is spelled four ways across the library (verified against the 2025 Q4 files on
2026-09-18):

| Format | Example | Where it appears |
|---|---|---|
| **components** (three columns) | `COMPANY ID = AAF`, `MAIN TYPE = N`, `SUB TYPE = 0000` (or `0`, or `0.0` once pandas has read it) | Daily Shares Price Lists 2020–2025, Foreign Holding |
| **hyphen** | `AAF-N-0000` (also `AAF -R-0000` with a stray space in the 2014 sheet, and `TILE-N-0` in Dividends 2010) | List of Quoted Securities And Issued Quantity, Dividends |
| **dot** | `AAF.N0000` (one lower-case `VLL.x0000` exists) | Public Holding, Market Capitalisation, Block Trades, Sub Division 2025 |
| **bare** | `JKH` | Sub Division 2011–2024 |

Two other things appear in identifier columns and are deliberately **rejected** as malformed:
debenture codes (`AAF/BD/18/12/27`) and joint entries (`COMB-N-0000 & COMB-X-0000`, Dividends).
A company name (`Unisyst`, Splits 2021) is also rejected.

## Canonical form

```python
SecurityId(company_code="AAF", share_class="N", sub_type="0000")
str(sid) == sid.canonical == "AAF-N-0000"
```

**Why the hyphen form `COMPANY-CLASS-SUBTYPE`:**

1. It is the CSE's own official *Security ID*, the form printed in the exchange's master list
   ("List of Quoted Securities And Issued Quantity"), which is also the only file carrying ISINs. Using
   the exchange's identifier keeps the framework traceable to the authoritative record (Chapter 4 §4.7:
   "official exchange data is preferred").
2. It is the only format that separates all three components with a visible delimiter, so it can be
   split without positional rules and is unambiguous for a human reader.
3. It round-trips losslessly to every other format, and sorts sensibly.
4. The sub type is always zero-padded to four digits, removing the `0000` / `0` / `0.0` variation.

The in-memory canonical object is a frozen dataclass, so it is hashable and works as a dict key or
pandas merge key.

## Share classes are distinct securities

`N` (voting ordinary), `X` (non-voting ordinary) and `P` (preference) trade separately with different
prices. **They are never collapsed.** `COMB-N-0000 != COMB-X-0000` and they hash differently.

Classes observed in the library and accepted by the parser: `N`, `X`, `P`, `R` (rights), `W`
(warrants), `U` (units), and `B`, `S` (seen only in the quoted-securities master; meaning not
documented there). Any other letter raises `UnknownShareClass` rather than being accepted silently.
Sub types other than `0000` exist for preference shares (`0001`–`0005`), rights (`0001`) and warrants
(up to `0024`); they are preserved.

## Unknown share class: never guessed

A bare ticker (`JKH`) or a row with a blank MAIN TYPE returns an `UnresolvedSecurityId`, not a
`SecurityId`:

```python
u = parse("JKH")
u.is_resolved      # False
u.canonical        # None
u.company_code     # "JKH"
to_dot(u)          # TypeError — refuses to invent a class
u.resolve("N")     # SecurityId("JKH", "N", "0000") once the caller has decided
```

The caller decides how to resolve it (for splits, the expected rule is: look the code up in the
quoted-securities master for that year; if the company has both `N` and `X` lines, the split applies to
both, as the `CIC.N0000 (N & X)` row in Splits 2025 shows). That lookup belongs in the corporate-action
loader, not here.

## API

```python
from identifiers import parse, parse_components, parse_hyphen, parse_dot, parse_bare
from identifiers import to_components, to_hyphen, to_dot, to_bare
from identifiers import SecurityId, UnresolvedSecurityId, MalformedIdentifier, UnknownShareClass

parse("AAF.N0000")           # SecurityId AAF-N-0000   (dot, auto-detected)
parse("AAF-N-0000")          # SecurityId AAF-N-0000   (hyphen)
parse(("AAF", "N", "0"))     # SecurityId AAF-N-0000   (components as a 3-tuple)
parse("AAF")                 # UnresolvedSecurityId    (bare)
parse("AAF/BD/18/12/27")     # raises MalformedIdentifier

to_components(sid)           # ("AAF", "N", "0000")
to_hyphen(sid)               # "AAF-N-0000"
to_dot(sid)                  # "AAF.N0000"
to_bare(sid)                 # "AAF"   (lossy)
```

Whitespace and case are normalised on input (`"  aaf.n0000 "` → `AAF-N-0000`). Output is always
upper case with no whitespace.

## Out of scope: name matching

"Capitalization of Reserves (Bonus Issues)", "Rights Issues" and "Scrip Dividends" identify companies by
**name only** (`Tokyo Cement Company (Lanka) PLC - Voting`, `HNB Finance PLC (Non - Voting)`), with
spelling that varies between files (`Softlogic Life` vs `SOFTLOGIC LIFE INSURANCE PLC`, double spaces in
`WATAWALA  PLANTATIONS  PLC`). Mapping names to company codes is a separate concern (a name→code
lookup table built from the quoted-securities master plus a manually reviewed alias list) and is not
attempted in this module. Chapter 4 does not specify how it is done, so per CLAUDE.md §0 the approach
must be agreed with the author before it is built.

## Running the tests

```bash
.venv/Scripts/python.exe -m pytest tests -q
```

Python 3.12 (Table 4.7). Configuration is in `pyproject.toml` (`pythonpath = ["src"]`).
