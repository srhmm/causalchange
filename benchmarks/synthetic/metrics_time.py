from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


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



@dataclass(frozen=True)
class PartitionMetrics:
    ari: float
    ami: float
    nmi: float


@dataclass(frozen=True)
class TargetPartitionMetrics:
    ari_mean: float
    ami_mean: float
    nmi_mean: float
    ari_by_target: dict[str, float]
    ami_by_target: dict[str, float]
    nmi_by_target: dict[str, float]


def compute_partition_metrics(
    true_labels: Sequence[int],
    estimated_labels: Sequence[int],
) -> PartitionMetrics:
    if len(true_labels) != len(estimated_labels):
        raise ValueError(f"Partition lengths differ: true={len(true_labels)}, estimated={len(estimated_labels)}.")

    if len(true_labels) == 0:
        return PartitionMetrics(
            ari=float("nan"),
            ami=float("nan"),
            nmi=float("nan"),
        )

    return PartitionMetrics(
        ari=float(adjusted_rand_score(true_labels, estimated_labels)),
        ami=float(adjusted_mutual_info_score(true_labels, estimated_labels)),
        nmi=float(normalized_mutual_info_score(true_labels, estimated_labels)),
    )


def compute_target_partition_metrics(
    true_labels_by_target: Mapping[str, Mapping[int, int]],
    estimated_labels_by_target: Mapping[str, Mapping[int, int]],
) -> TargetPartitionMetrics:
    ari_by_target: dict[str, float] = {}
    ami_by_target: dict[str, float] = {}
    nmi_by_target: dict[str, float] = {}

    for target, true_labels_map in true_labels_by_target.items():
        estimated_labels_map = estimated_labels_by_target[target]

        ids = sorted(true_labels_map.keys())
        true_labels = [int(true_labels_map[idx]) for idx in ids]
        estimated_labels = [int(estimated_labels_map[idx]) for idx in ids]

        scores = compute_partition_metrics(true_labels, estimated_labels)

        ari_by_target[target] = scores.ari
        ami_by_target[target] = scores.ami
        nmi_by_target[target] = scores.nmi

    return TargetPartitionMetrics(
        ari_mean=float(np.nanmean(list(ari_by_target.values()))) if ari_by_target else float("nan"),
        ami_mean=float(np.nanmean(list(ami_by_target.values()))) if ami_by_target else float("nan"),
        nmi_mean=float(np.nanmean(list(nmi_by_target.values()))) if nmi_by_target else float("nan"),
        ari_by_target=ari_by_target,
        ami_by_target=ami_by_target,
        nmi_by_target=nmi_by_target,
    )

def _changepoints_to_intervals(
    n_samples: int,
    changepoints: list[int],
) -> list[tuple[int, int]]:
    cps = sorted(int(cp) for cp in changepoints if 0 < int(cp) < n_samples)
    boundaries = [0, *cps, int(n_samples)]
    return [
        (int(boundaries[i]), int(boundaries[i + 1]))
        for i in range(len(boundaries) - 1)
    ]


def _expand_interval_labels_to_time(
    labels_by_interval: Mapping[int, int],
    changepoints: list[int],
    *,
    n_samples: int,
) -> list[int]:
    intervals = _changepoints_to_intervals(n_samples, changepoints)
    labels = [0 for _ in range(n_samples)]

    for interval_id, (start, stop) in enumerate(intervals):
        label = int(labels_by_interval[interval_id])
        for t in range(start, stop):
            labels[t] = label

    return labels


def compute_target_regime_partition_metrics_over_time(
    true_labels_by_target: Mapping[str, Mapping[int, int]],
    true_changepoints: list[int],
    estimated_labels_by_target: Mapping[str, Mapping[int, int]],
    estimated_changepoints: list[int],
    *,
    n_samples: int,
) -> TargetPartitionMetrics:
    ari_by_target: dict[str, float] = {}
    ami_by_target: dict[str, float] = {}
    nmi_by_target: dict[str, float] = {}

    for target, true_labels_map in true_labels_by_target.items():
        estimated_labels_map = estimated_labels_by_target[target]

        true_labels = _expand_interval_labels_to_time(
            true_labels_map,
            true_changepoints,
            n_samples=n_samples,
        )
        estimated_labels = _expand_interval_labels_to_time(
            estimated_labels_map,
            estimated_changepoints,
            n_samples=n_samples,
        )

        scores = compute_partition_metrics(true_labels, estimated_labels)

        ari_by_target[target] = scores.ari
        ami_by_target[target] = scores.ami
        nmi_by_target[target] = scores.nmi

    return TargetPartitionMetrics(
        ari_mean=float(np.nanmean(list(ari_by_target.values()))) if ari_by_target else float("nan"),
        ami_mean=float(np.nanmean(list(ami_by_target.values()))) if ami_by_target else float("nan"),
        nmi_mean=float(np.nanmean(list(nmi_by_target.values()))) if nmi_by_target else float("nan"),
        ari_by_target=ari_by_target,
        ami_by_target=ami_by_target,
        nmi_by_target=nmi_by_target,
    )