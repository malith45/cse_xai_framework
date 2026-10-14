# Explainable ML Framework for Stock Movement and Risk Prediction on the Colombo Stock Exchange

Final-year dissertation artifact (BSc (Hons) Computing, Coventry University via NIBM), built with the
Design Science Research methodology (Peffers et al., 2007). Decision support only: no automated
trading and no financial advice.

The framework predicts the next-day (t+1) and five-day (t+5) direction of S&P Sri Lanka 20
constituents, explains every prediction with SHAP, scores English, Sinhala and Tamil news, and
estimates Value-at-Risk with a GJR-GARCH(1,1)-t Monte Carlo engine validated by the Kupiec test.

## Repository layout

| Folder | Module (Chapter 4, Table 4.3) | What it contains |
|---|---|---|
| `pipeline/src/identifiers`, `readers`, `loaders` | M1 | CSE security-ID normaliser and robust readers for the CSE Data Library workbooks |
| `pipeline/src/pipeline` | M1, M2 | Price loader (incl. the 2023 CSV repair), corporate-action adjustment, trading calendar, point-in-time features and labels |
| `pipeline/src/sentiment`, `pipeline/scraper`, `pipeline/scripts/kaggle_sentiment.py` | M3 | News collector, language routing, alias dictionary, FinBERT / XLM-RoBERTa / NLLB scoring |
| `pipeline/src/models` | M4, M5 | Walk-forward splitter, XGBoost, LSTM/GRU, baselines, TreeSHAP / expected-gradient SHAP, narratives |
| `risk_engine/` | M6 | GJR-GARCH(1,1)-t, Monte Carlo VaR/ES, stress tests, Kupiec test (own test suite) |
| `pipeline/scripts/evaluate.py`, `analysis/` | M7 | Metrics per block, baseline comparison, McNemar tests, questionnaire and user-evaluation analysis |
| `dashboard/` | M8 | Five-page Streamlit dashboard (passcode, disclaimer on every page) |
| `pipeline/tests/` | NFR07 | Reader, identifier and look-ahead leakage tests |

## Data

The CSE Data Library is a paid, licence-restricted dataset and is **not** in this repository. Point
`CSE_DATA_ROOT` at a local copy. The public macro series (CBSL, Department of Census and Statistics)
are included in `pipeline/data_sources/macro/` with their sources and availability rule. The S&P SL20 point-in-time membership is rebuilt from the CSE
announcements by `pipeline/data_sources/build_membership.py` (sources listed in the script).

## Reproducing the results

```bash
cd pipeline
pip install -r requirements.txt
export CSE_DATA_ROOT=".../CSE Docs/All Data Files/2025 Q4-.../2025 Q4"
python scripts/build_panel.py --from-raw        # M1 + M2: adjusted prices, macro (data_sources/macro), feature panel
python -m pytest tests -q                       # readers, identifiers, leakage tests (incl. macro availability)
python scripts/run_xgb_baselines.py --feature-set prices --skip-baselines   # ablation: prices only
python scripts/run_xgb_baselines.py --feature-set macro                    # M4/M5 XGBoost + baselines
python scripts/select_feature_set.py            # deployed set per horizon, chosen on inner validation
python scripts/run_risk.py                      # M6 rolling VaR + Kupiec
python scripts/evaluate.py                      # M7 -> work/results/evaluation.xlsx
python scripts/publish_dashboard.py             # dashboard tables
```

LSTM/GRU training and the transformer sentiment models run on a Kaggle GPU notebook
(`scripts/run_rnn.py`, `scripts/kaggle_sentiment.py`). Fixed seeds and pinned versions (NFR09).
Implementation decisions that refine Chapter 4 are logged in `DECISIONS.md`.
