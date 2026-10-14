# Implementation decisions log

Decisions taken while building the artifact that refine or deviate from Chapter 4. Each one needs to
appear in the thesis (Chapter 4 design text or Chapter 5 limitations). Earlier decisions (2026-09-18,
2026-09-19) are in CLAUDE.md §7.

| Date | Topic | Decision | Reason | Thesis impact |
|---|---|---|---|---|
| 2026-10-13 | 2023 price CSV defect | Rebuild split numbers from fragments (`thousands x 1000 + remainder`), accept a row only if exactly one reading passes all checks (volume integers, high >= low, prices within x1.5, turnover/volume inside [low, high]) | Author's choice; deterministic and checkable | Ch 4 §4.7 step 1; Ch 5: 1,137 rows affected, 1,136 rebuilt, 1 dropped (CTC 2023-08-17) |
| 2026-10-13 | Scrip dividends | Not adjusted separately; the cash-dividend file's rate already includes the scrip part and shares its ex-date | Avoid double counting | Ch 4 §4.7 step 4 |
| 2026-10-13 | Bonus and rights ex-dates | Files give allotment dates only; ex-date taken as the day of the matching price drop within [allotment - 20, allotment + 5] trading days, else the allotment date | Ex-dates not in the library | Ch 4 §4.7 step 4 |
| 2026-10-13 | Unchanged prices | |log return| <= 1e-10 treated as exactly 0 (label 0 = "not up") | Float noise from adjustment factors | Ch 4 §4.7 step 7 |
| 2026-10-13 | Selection metric | Macro-F1 (mean of up and down F1) for Optuna tuning, threshold choice and LSTM-vs-GRU selection; positive-class F1 also reported | Positive-class F1 is maximised by always predicting "up" | Ch 4 §4.6.1-4.6.2 "F1-based" -> "macro-averaged F1" |
| 2026-10-13 | Tamil news source | Hiru News Tamil edition replaces Virakesari | Virakesari site did not respond to automated access; Hiru publishes EN/SI/TA with the same sitemap structure | Ch 4 §4.7, Table 4.5, FR04 |
| 2026-10-13 | NewsData.io | Not used; free plan returns only the past 48 hours and the archive endpoint is paid | Study period 2023-2025 is unreachable on the free plan | Ch 4 FR04, Table 4.5, §4.7; Ch 6 future work (live collection) |
| 2026-10-13 | News period | 2023-01-01 to 2025-12-31 (3 years); sentiment features are missing before 2023 (XGBoost handles missing values natively) | Collection time; covers every walk-forward test block (first test day 2023-04-26) | Ch 4 §4.7; Ch 5 sentiment ablation |
| 2026-10-13 | News without a time of day | Assigned to the next trading day (treated as after the 14:30 close) | Conservative: prevents look-ahead | Ch 4 §4.6.4 |
| 2026-10-13 | NLLB fallback score | When the fallback fires, the FinBERT score of the translation is the score used downstream; the XLM-R score is stored too | Purpose of the fallback is to replace a low-confidence score | Ch 4 §4.6.4 |
| 2026-10-13 | GJR-GARCH fits at the persistence bound | Accepted and flagged (`strict=False`) instead of carrying the previous day's parameters; counts reported per stock | Carrying stale parameters affected up to ~36% of days for some stocks; 1-5 day VaR remains well defined | Ch 4 §4.6.5; Ch 5 risk limitations |
| 2026-10-13 | Dashboard crisis scenario | Volatility multiplier = sqrt(2022-fit unconditional variance / current variance), clipped to [1, 4] (2.0 when undefined); shock = 1st percentile of the stock's 2022 daily returns | Concrete calibration of "2022-crisis scenario" | Ch 4 §4.6.5 |
| 2026-10-13 | Compute environment | Offline pipeline run in a cloud Linux workspace (Python 3.13); LSTM/GRU and transformers on Kaggle GPU | Time constraint; CPU too slow for RNN tuning | Ch 4 Table 4.8 |
| 2026-10-13 | Dashboard sector field | Shown as "—" | Company-level GICS sector is not in the Data Library | Ch 4 §4.5.5 UI |
