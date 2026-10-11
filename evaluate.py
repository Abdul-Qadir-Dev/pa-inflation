"""Point-wise F1, point-adjusted F1 and AUROC."""
import numpy as np
from sklearn.metrics import roc_auc_score

N_THRESHOLDS = 200


def segments(y):
    """Start/end (exclusive) of each run of 1s in y."""
    d = np.diff(np.concatenate([[0], y, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def point_adjust(pred, y, segs=None):
    # the usual adjustment: one hit anywhere in a labelled segment marks the whole segment as found
    pred = pred.copy()
    for s, e in segs if segs is not None else segments(y):
        if pred[s:e].any():
            pred[s:e] = 1
    return pred


def f1(pred, y):
    tp = np.sum(pred & y)
    if tp == 0:
        return 0.0
    precision = tp / pred.sum()
    recall = tp / y.sum()
    return 2 * precision * recall / (precision + recall)


def thresholds(scores):
    return np.unique(np.quantile(scores, np.linspace(0, 1, N_THRESHOLDS)))


def evaluate(scores, y):
    y = y.astype(bool)
    segs = segments(y.astype(int))
    best_f1, best_pa = 0.0, 0.0
    # best F1 and best PA-F1 can come from different thresholds, as in the papers that report them
    for t in thresholds(scores):
        pred = scores >= t
        best_f1 = max(best_f1, f1(pred, y))
        best_pa = max(best_pa, f1(point_adjust(pred, y, segs), y))

    # TODO: add PA%K from Kim et al. (only adjust a segment if at least K% of it is flagged)

    # sanity line: a fixed 1% flag rate, no peeking at labels to pick the threshold
    pred = scores >= np.quantile(scores, 0.99)
    return {
        "f1": best_f1,
        "pa_f1": best_pa,
        "f1_at1": f1(pred, y),
        "pa_f1_at1": f1(point_adjust(pred, y, segs), y),
        "auroc": roc_auc_score(y, scores),
    }
