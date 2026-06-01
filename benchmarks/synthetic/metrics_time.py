from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChangepointMetrics:
    changepoint_precision: float
    changepoint_recall: float
    changepoint_f1: float
    changepoint_mean_abs_error: float
    changepoint_n_true: int
    changepoint_n_est: int


def compute_changepoint_metrics(
    true_changepoints: Iterable[int],
    estimated_changepoints: Iterable[int],
    *,
    tolerance: int = 5,
) -> ChangepointMetrics:
    true_cps = sorted(int(cp) for cp in true_changepoints)
    est_cps = sorted(int(cp) for cp in estimated_changepoints)

    candidate_matches: list[tuple[int, int, int]] = []
    for true_idx, true_cp in enumerate(true_cps):
        for est_idx, est_cp in enumerate(est_cps):
            distance = abs(true_cp - est_cp)
            if distance <= tolerance:
                candidate_matches.append((distance, true_idx, est_idx))

    matched_true: set[int] = set()
    matched_est: set[int] = set()
    abs_errors: list[int] = []

    for distance, true_idx, est_idx in sorted(candidate_matches):
        if true_idx in matched_true or est_idx in matched_est:
            continue

        matched_true.add(true_idx)
        matched_est.add(est_idx)
        abs_errors.append(distance)

    tp = len(matched_true)
    fp = len(est_cps) - tp
    fn = len(true_cps) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else float(fn == 0)
    recall = tp / (tp + fn) if tp + fn > 0 else float(fp == 0)

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    mean_abs_error = float(np.mean(abs_errors)) if abs_errors else float("nan")

    return ChangepointMetrics(
        changepoint_precision=float(precision),
        changepoint_recall=float(recall),
        changepoint_f1=float(f1),
        changepoint_mean_abs_error=mean_abs_error,
        changepoint_n_true=len(true_cps),
        changepoint_n_est=len(est_cps),
    )
