"""M4 LSTM/GRU sequence classifier (§4.6.2) and expected-gradient SHAP (§4.6.3). PyTorch only (D3).

Each sample is the 30 most recent *traded* days (up to and including day t) of one security's
standardised features; the target is that security's label on day t. Scaling is fitted per security
on the training window only. Both LSTM and GRU are tuned per block with Optuna (15 trials), trained
with BCE loss, Adam, dropout, class weights and early stopping on the inner-validation slice. The
architecture with the higher mean inner-validation score across blocks is deployed (decision D5);
test blocks are never used for selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn

from . import config as C
from .walkforward import PerStockScaler, best_threshold, f1_macro, in_range

torch.set_num_threads(2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeqNet(nn.Module):
    def __init__(self, arch: str, n_in: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        rnn = nn.LSTM if arch == "lstm" else nn.GRU
        self.rnn = rnn(n_in, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


def build_sequences(df: pd.DataFrame, scaler: PerStockScaler, features: list[str], label: str,
                    lookback: int = C.RNN_LOOKBACK) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Sequences for every row of ``df`` that has ``lookback`` rows of history *within df*.

    ``df`` must contain each security's full history up to the latest target day (callers pass all
    rows with cal_pos < end of the slice), so windows can reach back before the slice start."""
    Z = scaler.transform(df).fillna(0.0).to_numpy(np.float32)
    X, y, keys = [], [], []
    for sid, idx in df.groupby("security_id").indices.items():
        idx = np.sort(idx)
        sub = df.iloc[idx]
        order = np.argsort(sub["cal_pos"].values)
        idx = idx[order]
        z = Z[idx]
        lab = df[label].values[idx]
        for j in range(lookback - 1, len(idx)):
            X.append(z[j - lookback + 1: j + 1])
            y.append(lab[j])
            keys.append(idx[j])
    if not X:
        return np.empty((0, lookback, len(features)), np.float32), np.empty(0), df.iloc[:0]
    return np.stack(X), np.asarray(y, dtype=np.float32), df.iloc[keys]


def _subset(X, y, k, mask):
    m = mask & ~np.isnan(y)
    return X[m], y[m], k[m]


def _train(arch, params, Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = SeqNet(arch, Xtr.shape[2], params["hidden"], params["layers"], params["dropout"]).to(DEVICE)
    pos = ytr.mean()
    pw = torch.tensor((1 - pos) / max(pos, 1e-6), device=DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(net.parameters(), lr=params["lr"])
    Xt, yt = torch.from_numpy(Xtr).to(DEVICE), torch.from_numpy(ytr).to(DEVICE)
    Xv = torch.from_numpy(Xva).to(DEVICE)
    best, best_state, bad = -np.inf, None, 0
    bs = 256
    for ep in range(C.RNN_MAX_EPOCHS):
        net.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv)).cpu().numpy()
        thr = best_threshold(yva, pv)
        score = f1_macro(yva, (pv >= thr).astype(int))
        if score > best + 1e-4:
            best, best_state, best_thr, bad = score, {k: v.detach().clone() for k, v in net.state_dict().items()}, thr, 0
        else:
            bad += 1
            if bad >= C.RNN_PATIENCE:
                break
    net.load_state_dict(best_state)
    return net, best, best_thr


@dataclass
class RNNResult:
    arch: str
    net: SeqNet
    scaler: PerStockScaler
    threshold: float
    params: dict
    inner_val_score: float
    background: np.ndarray


def train_rnn_block(panel: pd.DataFrame, features: list[str], label: str, blk, arch: str,
                    n_trials: int = C.RNN_TRIALS) -> RNNResult:
    hist = panel[panel["cal_pos"] < blk.train[1]]
    mem = hist["member"] if "member" in hist else pd.Series(True, index=hist.index)
    sc = PerStockScaler(features).fit(hist[in_range(hist["cal_pos"], blk.inner_train) & hist[label].notna() & mem])
    X, y, k = build_sequences(hist, sc, features, label)
    pos = k["cal_pos"].values
    km = k["member"].values.astype(bool) if "member" in k else np.ones(len(k), bool)
    # targets only on days the security was a constituent; look-back windows may reach before it joined
    Xtr, ytr, _ = _subset(X, y, k, km & (pos >= blk.inner_train[0]) & (pos < blk.inner_train[1]))
    Xva, yva, _ = _subset(X, y, k, km & (pos >= blk.inner_val[0]) & (pos < blk.inner_val[1]))
    store = {}

    def objective(trial):
        params = {
            "layers": trial.suggest_categorical("layers", list(C.RNN_LAYERS)),
            "hidden": trial.suggest_categorical("hidden", list(C.RNN_HIDDEN)),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        }
        net, score, thr = _train(arch, params, Xtr, ytr, Xva, yva, C.SEED + blk.b)
        store[trial.number] = (net, thr, params)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED + blk.b))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    net, thr, params = store[study.best_trial.number]
    rng = np.random.default_rng(C.SEED)
    bg = Xtr[rng.choice(len(Xtr), size=min(C.SHAP_BACKGROUND, len(Xtr)), replace=False)]
    return RNNResult(arch, net, sc, thr, params, study.best_value, bg)


def predict_rnn(res: RNNResult, panel: pd.DataFrame, features: list[str], label: str, blk):
    hist = panel[panel["cal_pos"] < blk.test[1]]
    X, y, k = build_sequences(hist, res.scaler, features, label)
    m = (k["cal_pos"].values >= blk.test[0]) & (k["cal_pos"].values < blk.test[1])
    if "member" in k:
        m &= k["member"].values.astype(bool)
    X, k = X[m], k[m]
    res.net.eval()
    with torch.no_grad():
        p = torch.sigmoid(res.net(torch.from_numpy(X).to(DEVICE))).cpu().numpy() if len(X) else np.empty(0)
    return p, X, k


def gradient_shap(res: RNNResult, X: np.ndarray, nsamples: int = 50) -> np.ndarray:
    """Expected-gradients attributions (shap.GradientExplainer) summed over the 30 time steps ->
    [n, n_features], in logit units."""
    import shap

    class Logit(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x).unsqueeze(-1)

    res.net.train(False)   # eval mode: no dropout; RNN backward in eval mode is supported on CPU
    model = Logit(res.net)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.enabled = False   # cuDNN RNN backward requires train mode
    exp = shap.GradientExplainer(model, torch.from_numpy(res.background).to(DEVICE))
    out = []
    for i in range(0, len(X), 512):
        sv = exp.shap_values(torch.from_numpy(X[i:i + 512]).to(DEVICE), nsamples=nsamples)
        sv = sv[0] if isinstance(sv, list) else sv
        sv = np.asarray(sv)
        if sv.ndim == 4:
            sv = sv[..., 0]
        out.append(sv.sum(axis=1))
    torch.backends.cudnn.enabled = True
    return np.concatenate(out) if out else np.empty((0, X.shape[2]))
