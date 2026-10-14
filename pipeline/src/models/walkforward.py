"""M4 walk-forward splitter, per-stock scaling and the XGBoost directional classifier (§4.6.1).

Blocks (Figure 4.4). With trading-calendar positions 0..N-1:

    test_start_b = INITIAL_TRAIN_DAYS + PURGE_DAYS + b * TEST_BLOCK_DAYS
    test         = [test_start_b, test_start_b + TEST_BLOCK_DAYS)
    train        = [0, test_start_b - PURGE_DAYS)               (expanding; 750 days in block 0)
    inner_val    = last INNER_VAL_DAYS of train
    inner_train  = train before inner_val, minus another PURGE_DAYS gap

The purge drops the last 5 training days before each evaluation slice, so no training label
(which looks up to 5 days ahead) overlaps the slice it is evaluated on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

from . import config as C

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class Block:
    b: int
    train: tuple[int, int]
    inner_train: tuple[int, int]
    inner_val: tuple[int, int]
    test: tuple[int, int]


def make_blocks(n_days: int) -> list[Block]:
    blocks, b = [], 0
    while True:
        ts = C.INITIAL_TRAIN_DAYS + C.PURGE_DAYS + b * C.TEST_BLOCK_DAYS
        if ts >= n_days:
            break
        te = min(ts + C.TEST_BLOCK_DAYS, n_days)
        tr_end = ts - C.PURGE_DAYS
        iv = (tr_end - C.INNER_VAL_DAYS, tr_end)
        it = (0, iv[0] - C.PURGE_DAYS)
        blocks.append(Block(b, (0, tr_end), it, iv, (ts, te)))
        b += 1
    return blocks


def in_range(pos: pd.Series, r: tuple[int, int]) -> pd.Series:
    return (pos >= r[0]) & (pos < r[1])


# --------------------------------------------------------------------------- scaling (step 11)

class PerStockScaler:
    """z-score each feature per security using statistics from the fitting rows only."""

    def __init__(self, features: list[str]):
        self.features = features

    def fit(self, df: pd.DataFrame) -> "PerStockScaler":
        g = df.groupby("security_id")[self.features]
        self.mu, self.sd = g.mean(), g.std().replace(0, np.nan)
        self.gmu, self.gsd = df[self.features].mean(), df[self.features].std().replace(0, np.nan)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        mu = self.mu.reindex(df["security_id"]).fillna(self.gmu)
        sd = self.sd.reindex(df["security_id"]).fillna(self.gsd)
        z = (df[self.features].values - mu.values) / sd.values
        return pd.DataFrame(np.clip(z, -10, 10), index=df.index, columns=self.features)


# --------------------------------------------------------------------------- metrics helpers

def f1_macro(y, p) -> float:
    return f1_score(y, p, average="macro", zero_division=0)


def best_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    grid = np.quantile(prob, np.linspace(0.05, 0.95, 37))
    scores = [f1_macro(y, (prob >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


# --------------------------------------------------------------------------- XGBoost

def _design(df: pd.DataFrame, scaler: PerStockScaler, cats: list[str]) -> pd.DataFrame:
    X = scaler.transform(df)
    X["stock"] = pd.Categorical(df["security_id"].values, categories=cats)
    return X


def _xgb_params(trial: optuna.Trial, spw: float) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "enable_categorical": True,
        "max_cat_to_onehot": 1,
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "scale_pos_weight": spw,
        "random_state": C.SEED,
        "n_jobs": int(__import__("os").environ.get("XGB_THREADS", 2)),
    }


@dataclass
class XGBResult:
    model: xgb.XGBClassifier
    scaler: PerStockScaler
    threshold: float
    params: dict
    inner_val_score: float


def train_xgb_block(panel: pd.DataFrame, features: list[str], label: str, blk: Block, cats: list[str],
                    n_trials: int = C.XGB_TRIALS) -> XGBResult:
    pos = panel["cal_pos"]
    d_it = panel[in_range(pos, blk.inner_train) & panel[label].notna()]
    d_iv = panel[in_range(pos, blk.inner_val) & panel[label].notna()]
    d_tr = panel[in_range(pos, blk.train) & panel[label].notna()]

    sc_inner = PerStockScaler(features).fit(d_it)
    X_it, y_it = _design(d_it, sc_inner, cats), d_it[label].astype(int).values
    X_iv, y_iv = _design(d_iv, sc_inner, cats), d_iv[label].astype(int).values
    spw = float((y_it == 0).sum() / max(1, (y_it == 1).sum()))

    def objective(trial):
        m = xgb.XGBClassifier(**_xgb_params(trial, spw))
        m.fit(X_it, y_it, verbose=False)
        prob = m.predict_proba(X_iv)[:, 1]
        t = best_threshold(y_iv, prob)
        trial.set_user_attr("threshold", t)
        return f1_macro(y_iv, (prob >= t).astype(int))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED + blk.b))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    params = _xgb_params(optuna.trial.FixedTrial(best.params), spw)

    # refit on the full training window (inner-train + purge + inner-val) with the chosen params
    sc = PerStockScaler(features).fit(d_tr)
    X_tr, y_tr = _design(d_tr, sc, cats), d_tr[label].astype(int).values
    params["scale_pos_weight"] = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, verbose=False)
    return XGBResult(model, sc, best.user_attrs["threshold"], best.params, best.value)


def predict_xgb(res: XGBResult, df: pd.DataFrame, cats: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    X = _design(df, res.scaler, cats)
    return res.model.predict_proba(X)[:, 1], X
