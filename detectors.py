"""Detectors. Each takes scaled train and test arrays and returns one score per test point (higher = more anomalous)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from windows import make_windows, test_windows

TRAIN_STRIDE = 8  # training windows only; scoring always uses stride 1


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


def random_scores(train, test, seed=0):
    return np.random.default_rng(seed).uniform(size=len(test))


DETECTORS = {
    "zscore": zscore,
    "iforest": iforest,
    "random": random_scores,
}
