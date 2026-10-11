"""Loading series, train/test split, scaling and sliding windows."""
import json
import os

import numpy as np
import pandas as pd

DATA = "data"
NAB_DIRS = ["realAWSCloudwatch", "realKnownCause", "realTweets"]
WIN = 64


def nab_names():
    with open(os.path.join(DATA, "nab", "combined_windows.json")) as f:
        windows = json.load(f)
    names = []
    for k in sorted(windows):
        if k.split("/")[0] in NAB_DIRS and os.path.exists(os.path.join(DATA, "nab", k)):
            names.append(k)
    return names, windows


def load_nab(name, windows):
    """Returns values and 0/1 labels. A point is anomalous if it falls inside any labelled window."""
    df = pd.read_csv(os.path.join(DATA, "nab", name), parse_dates=["timestamp"])
    y = np.zeros(len(df), dtype=int)
    for start, end in windows[name]:
        y[((df.timestamp >= pd.Timestamp(start)) & (df.timestamp <= pd.Timestamp(end))).values] = 1
    return df.value.values.astype(float), y


def nab_series():
    """Yields (name, train_x, test_x, test_y). NAB has no clean train split, so the first half is train."""
    names, windows = nab_names()
    for name in names:
        x, y = load_nab(name, windows)
        half = len(x) // 2
        yield name, x[:half], x[half:], y[half:]


def nasa_series():
    """Yields the same tuple for SMAP/MSL, using the release's own train/test files.

    Only channel 0 (the telemetry value) is used; the other columns are one-hot command flags.
    """
    root = os.path.join(DATA, "nasa")
    if not os.path.exists(os.path.join(root, "test")):
        return
    labels = pd.read_csv(os.path.join(root, "labeled_anomalies.csv"))
    for _, row in labels.iterrows():
        chan = row.chan_id
        train = np.load(os.path.join(root, "train", chan + ".npy"))[:, 0]
        test = np.load(os.path.join(root, "test", chan + ".npy"))[:, 0]
        y = np.zeros(len(test), dtype=int)
        for s, e in json.loads(row.anomaly_sequences):
            y[s:e + 1] = 1
        yield f"{row.spacecraft}/{chan}", train.astype(float), test.astype(float), y


def minmax(train, test):
    # fit on train only; test values can land outside [0, 1] and that's fine
    lo, hi = train.min(), train.max()
    span = hi - lo if hi > lo else 1.0
    return (train - lo) / span, (test - lo) / span


def make_windows(x, stride=1, win=WIN):
    idx = np.arange(0, len(x) - win + 1, stride)
    return np.stack([x[i:i + win] for i in idx])


def test_windows(train, test, win=WIN):
    """One window per test point, ending at that point.

    The first win-1 test points borrow context from the end of the train portion,
    so every test point gets a score and the score array lines up with test labels.
    """
    ctx = np.concatenate([train[-(win - 1):], test])
    w = make_windows(ctx, stride=1, win=win)
    assert len(w) == len(test)
    return w


def all_series(datasets=("nab", "nasa")):
    for ds in datasets:
        gen = nab_series() if ds == "nab" else nasa_series()
        for name, train, test, y in gen:
            yield ds, name, train, test, y
