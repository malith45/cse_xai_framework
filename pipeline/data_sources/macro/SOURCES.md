# Macro series (Chapter 4 §4.7 step 9) — collected 2026-10-14

All values copied from official public pages through the author's browser; each CSV's SHA-256 was
computed in the browser and matched after transfer (cbsl_usd_lkr_daily, cbsl_tbill_weekly,
cbsl_awpr_weekly). Policy rates and CCPI were transcribed from the official files and cross-checked
(CCPI: every year-on-year figure re-derived from the index numbers; 0 mismatches).

| File | Series | Source | Frequency | Rows |
|---|---|---|---|---|
| cbsl_usd_lkr_daily.csv | Indicative USD/LKR spot rate (LKR per USD) | CBSL, cbsl.gov.lk/cbsl_custom/exrates/exrates_spot_mid.php | daily, 2019-06-03 to 2025-12-31 | 1,588 |
| cbsl_policy_rates.csv | SDFR / SLFR (OPR = midpoint from 27 Nov 2024) | CBSL, historical_policy_interest_rates.xlsx (linked from the Policy Rates page) | on change | 26 |
| cbsl_awpr_weekly.csv | Average Weighted Prime Lending Rate, weekly | CBSL Data Library (eResearch), Commercial Banks Lending and Deposit Rates | weekly | 343 |
| cbsl_tbill_weekly.csv | Treasury bill yields 91/182/364 days (end-week primary auction) | CBSL Data Library (eResearch), Rates on Government Securities | weekly | 340 |
| ccpi_monthly.csv | Colombo CPI index and year-on-year inflation | Department of Census and Statistics, "Movements of the CCPI" (base 2013=100 and 2021=100) | monthly | 84 |

Notes
- CCPI: year-on-year inflation is the base-2013 figure up to Dec 2022 and the base-2021 figure from
  Jan 2023 (DCS switched base in Jan 2023; the Jan 2023 cell is blank in the base-2021 table and is
  computed from the index, 51.7%, which equals the published headline).
- CCPI is released on the last working day of the reference month (web releases are named
  CCPI_YYYYMMDD, e.g. CCPI_20260930E for September 2026).
- T-bill rows contain "-" or blanks in weeks without an auction for that tenor.
