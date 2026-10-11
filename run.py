"""Score every series with every detector and write results/summary.csv and results/aggregate.md."""
import argparse
import os
import time

import numpy as np
import pandas as pd

from detectors import DETECTORS
from evaluate import evaluate
from windows import all_series, minmax

SEEDED = {"lstm_ae", "random"}  # the others are deterministic enough that one seed is fine
METRICS = ["f1", "pa_f1", "f1_at1", "pa_f1_at1", "auroc"]
# score traces for these get saved for the plots and the notebook
KEEP_SCORES = "realKnownCause/"


def run(datasets, detectors, n_seeds, limit):
    rows = []
    traces = {}
    count = {}
    for ds, name, train, test, y in all_series(datasets):
        if limit and count.get(ds, 0) >= limit:
            continue
        if y.sum() == 0:
            print(f"skip {ds}/{name}: no labelled anomaly in the test part")
            continue
        count[ds] = count.get(ds, 0) + 1
        train, test = minmax(train, test)
        for det in detectors:
            seeds = range(n_seeds) if det in SEEDED else [0]
            for seed in seeds:
                info = {}
                t0 = time.time()
                if det == "lstm_ae":
                    scores = DETECTORS[det](train, test, seed=seed, info=info)
                else:
                    scores = DETECTORS[det](train, test, seed=seed)
                row = {"dataset": ds, "detector": det, "seed": seed, "series": name,
                       "n_test": len(test), "anomaly_rate": y.mean(), "seconds": time.time() - t0}
                row.update(evaluate(scores, y))
                row.update(info)
                rows.append(row)
                if seed == 0 and name.startswith(KEEP_SCORES):
                    traces[f"{name}|{det}"] = scores.astype(np.float32)
                    traces[f"{name}|label"] = y.astype(np.int8)
            last = rows[-1]
            print(f"{ds} {name} {det}: f1={last['f1']:.3f} pa_f1={last['pa_f1']:.3f} auroc={last['auroc']:.3f}")
    return pd.DataFrame(rows), traces


def iqr_cell(s):
    return f"{s.median():.3f} [{s.quantile(0.25):.3f}, {s.quantile(0.75):.3f}]"


def aggregate(df):
    # average seeds within a series first, then median and IQR across series
    per_series = df.groupby(["dataset", "detector", "series"])[METRICS].mean().reset_index()
    order = ["random", "zscore", "iforest", "lstm_ae"]
    lines = []
    for ds, g in per_series.groupby("dataset"):
        lines.append(f"### {ds} ({g.series.nunique()} series)\n")
        lines.append("| detector | F1 | PA-F1 | F1 @1% | PA-F1 @1% | AUROC |")
        lines.append("|---|---|---|---|---|---|")
        for det in [d for d in order if d in set(g.detector)]:
            gd = g[g.detector == det]
            cells = [iqr_cell(gd[m]) for m in METRICS]
            lines.append(f"| {det} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("Median across series, interquartile range in brackets. "
                 "F1 and PA-F1 use the best of 200 thresholds per series; the @1% columns "
                 "use the threshold that flags 1% of test points. Seeds are averaged within a series first.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["nab", "nasa"])
    ap.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    ap.add_argument("--limit", type=int, default=0, help="only the first N series per dataset")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    df, traces = run(args.datasets, args.detectors, args.seeds, args.limit)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/summary.csv", index=False, float_format="%.4f")
    np.savez_compressed("results/traces.npz", **traces)
    table = aggregate(df)
    with open("results/aggregate.md", "w") as f:
        f.write(table)
    print()
    print(table)
