"""Design constants for M4/M5/M7. Every value cites its Chapter 4 source; change none without the
author's approval (CLAUDE.md §0)."""

SEED = 42                       # NFR09 fixed seeds

# Walk-forward (Figure 4.4, §4.5.3, §4.7 step 12)
INITIAL_TRAIN_DAYS = 750        # expanding window starts at 750 trading days
INNER_VAL_DAYS = 126            # last 126 trading days of each training window
PURGE_DAYS = 5                  # 5-trading-day purge/embargo
TEST_BLOCK_DAYS = 63            # 63-trading-day test blocks

# Tuning (§4.6.1, §4.6.2)
XGB_TRIALS = 30                 # Optuna TPE trials per block (XGBoost)
RNN_TRIALS = 15                 # Optuna TPE trials per block (LSTM/GRU)
RNN_LOOKBACK = 30               # 30-day lookback
RNN_LAYERS = (1, 2)             # 1-2 layers
RNN_HIDDEN = (32, 64)           # 32-64 hidden units
RNN_MAX_EPOCHS = int(__import__("os").environ.get("RNN_MAX_EPOCHS", 40))  # env override for smoke tests only
RNN_PATIENCE = 5                # early stopping on inner validation

# Selection metric. Chapter 4 says "F1-based"; positive-class F1 is maximised by predicting "up"
# for every row, so the macro-averaged F1 of the two classes is used for tuning, threshold choice
# and LSTM-vs-GRU selection (flagged to the author 2026-10-13). Positive-class F1 is still reported.
SELECTION_METRIC = "f1_macro"

HORIZONS = {"t1": "y1", "t5": "y5"}

# Explanations (§4.6.3)
SHAP_TOP_K = 10
SHAP_BACKGROUND = 200           # background sample drawn from the training window
