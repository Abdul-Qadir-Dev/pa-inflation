"""Figures from results/summary.csv and results/traces.npz."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import segments
from windows import load_nab, nab_names

EXAMPLE = "realKnownCause/machine_temperature_system_failure.csv"
ORDER = ["random", "zscore", "iforest", "lstm_ae"]
OUT = "plots"


def per_series():
    df = pd.read_csv("results/summary.csv")
    return df.groupby(["dataset", "detector", "series"])[["f1", "pa_f1", "auroc"]].mean().reset_index()


def example_series(traces):
    names, windows = nab_names()
    x, y = load_nab(EXAMPLE, windows)
    half = len(x) // 2
    for det in ORDER:
        scores = traces[f"{EXAMPLE}|{det}"]
        test, labels = x[half:], y[half:]
        t = np.arange(half, len(x))
        # 1% flag rate rather than the best-F1 threshold, so the picture doesn't use the labels
        flagged = scores >= np.quantile(scores, 0.99)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for s, e in segments(labels):
            ax1.axvspan(t[s], t[e - 1], color="orange", alpha=0.3)
            ax2.axvspan(t[s], t[e - 1], color="orange", alpha=0.3)
        ax1.plot(t, test, lw=0.7)
        ax1.scatter(t[flagged], test[flagged], color="red", s=8, zorder=3, label="flagged (top 1%)")
        ax1.set_ylabel("value")
        ax1.legend(loc="upper left")
        ax1.set_title(f"{det} on {EXAMPLE.split('/')[1][:-4]} (test half, labelled windows shaded)")
        ax2.plot(t, scores, lw=0.7, color="gray")
        ax2.set_ylabel("score")
        ax2.set_xlabel("time step")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"example_{det}.png"), dpi=110)
        plt.close(fig)


def bars(ps):
    datasets = sorted(ps.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets) + 2, 4.5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        g = ps[ps.dataset == ds]
        dets = [d for d in ORDER if d in set(g.detector)]
        xs = np.arange(len(dets))
        for off, metric, label in [(-0.2, "f1", "plain F1"), (0.2, "pa_f1", "point-adjusted F1")]:
            med = [g[g.detector == d][metric].median() for d in dets]
            lo = [m - g[g.detector == d][metric].quantile(0.25) for m, d in zip(med, dets)]
            hi = [g[g.detector == d][metric].quantile(0.75) - m for m, d in zip(med, dets)]
            ax.bar(xs + off, med, width=0.4, yerr=[lo, hi], capsize=3, label=label)
        ax.set_xticks(xs)
        ax.set_xticklabels(dets)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("median across series (bars: IQR)")
        ax.set_title(f"{ds.upper()}: plain vs point-adjusted F1")
        ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f1_vs_pa_f1_bars.png"), dpi=110)
    plt.close(fig)


def scatter(ps):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for det in ORDER:
        g = ps[ps.detector == det]
        if det == "random":
            ax.scatter(g.f1, g.pa_f1, s=60, color="black", marker="x", label="random (control)", zorder=3)
        else:
            ax.scatter(g.f1, g.pa_f1, s=22, alpha=0.7, label=det)
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("plain F1 (best threshold)")
    ax.set_ylabel("point-adjusted F1 (best threshold)")
    ax.set_title("Each point is one series; dotted line is y = x")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f1_vs_pa_f1_scatter.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ps = per_series()
    traces = np.load("results/traces.npz")
    example_series(traces)
    bars(ps)
    scatter(ps)
    print("wrote", sorted(os.listdir(OUT)))
