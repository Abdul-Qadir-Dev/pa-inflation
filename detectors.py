"""Detectors. Each takes scaled train and test arrays and returns one score per test point (higher = more anomalous)."""
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from torch import nn

from windows import make_windows, test_windows

TRAIN_STRIDE = 8  # training windows only; scoring always uses stride 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def zscore(train, test, seed=0, roll=100):
    # rolling stats over the previous `roll` points, so a spike doesn't inflate its own std
    x = pd.Series(np.concatenate([train, test]))
    mean = x.rolling(roll).mean().shift(1)
    std = x.rolling(roll).std().shift(1)
    z = ((x - mean) / std.replace(0, np.nan)).abs()
    return z.fillna(0).values[len(train):]


def iforest(train, test, seed=0):
    model = IsolationForest(n_estimators=100, random_state=seed)
    model.fit(make_windows(train, stride=TRAIN_STRIDE))
    # score_samples is higher for normal points, so flip it
    return -model.score_samples(test_windows(train, test))


class LSTMAE(nn.Module):
    """One-layer LSTM encoder, last hidden state repeated across time, one-layer LSTM decoder."""

    def __init__(self, hidden=32):
        super().__init__()
        self.enc = nn.LSTM(1, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.dec(z)
        return self.out(y)


def lstm_ae(train, test, seed=0, info=None, epochs=20, patience=3, cap=120, batch=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = DEVICE

    w = make_windows(train, stride=TRAIN_STRIDE)
    perm = rng.permutation(len(w))
    n_val = max(1, len(w) // 10)
    val = torch.tensor(w[perm[:n_val]], dtype=torch.float32, device=dev).unsqueeze(-1)
    tr = torch.tensor(w[perm[n_val:]], dtype=torch.float32, device=dev).unsqueeze(-1)

    model = LSTMAE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, bad = np.inf, None, 0
    t0 = time.time()
    cap_hit = False
    epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), batch):
            xb = tr[order[i:i + batch]]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = ((model(val) - val) ** 2).mean().item()
        if v < best:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if time.time() - t0 > cap:
            cap_hit = True
            break
    model.load_state_dict(best_state)

    # each window's error goes to its last point, same as the forest
    # TODO: try averaging the per-point error over every window that covers a point
    scores = []
    tw = test_windows(train, test)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tw), 2048):
            xb = torch.tensor(tw[i:i + 2048], dtype=torch.float32, device=dev).unsqueeze(-1)
            scores.append(((model(xb) - xb) ** 2).mean(dim=(1, 2)).cpu().numpy())
    if info is not None:
        info.update(epochs_run=epoch, train_seconds=time.time() - t0, cap_hit=cap_hit, val_mse=best)
    return np.concatenate(scores)


def random_scores(train, test, seed=0):
    return np.random.default_rng(seed).uniform(size=len(test))


DETECTORS = {
    "zscore": zscore,
    "iforest": iforest,
    "lstm_ae": lstm_ae,
    "random": random_scores,
}
