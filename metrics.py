# Copyright (c) 2026 Zhaoxin Yan.
"""Binary classification and probability scores."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Metrics:
    tn: int
    fp: int
    fn: int
    tp: int
    recall: float
    precision: float
    accuracy: float
    f1: float
    tss: float
    bs: float
    bss: float

    @property
    def acc(self):
        return self.accuracy


def compute_metrics(labels, probabilities, threshold=0.6):
    y = np.asarray(labels).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if not len(y) or y.shape != p.shape:
        raise ValueError('Labels and probabilities must have equal nonzero length.')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected binary labels and finite probabilities in [0, 1].')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0, 1].')
    pred = p >= threshold
    tn = int(((y == 0) & ~pred).sum()); fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum()); tp = int(((y == 1) & pred).sum())
    recall = tp / (tp + fn) if tp + fn else float('nan')
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    tss = recall - fp / (fp + tn) if fp + tn else float('nan')
    bs = float(np.mean((p - y) ** 2))
    reference = float(np.mean((y - y.mean()) ** 2))
    bss = 1 - bs / reference if reference else float('nan')
    return Metrics(tn, fp, fn, tp, recall, precision, (tp + tn) / len(y), f1, tss, bs, bss)
