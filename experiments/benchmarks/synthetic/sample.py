from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class MixedSyntheticResult:
    df: pd.DataFrame
    true_summary_dag: nx.DiGraph
    labels_by_target: dict[str, list[int]]
    mixed_targets: list[str]
    cluster_mode: str
    n_mechanisms: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MultiContextSyntheticResult:
    df: pd.DataFrame
    true_summary_dag: nx.DiGraph
    context_labels_by_target: dict[str, dict[int, int]]
    context_col: str
    variables: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SpaceTimeSyntheticResult:
    df: pd.DataFrame
    true_wcg: nx.DiGraph
    true_summary_dag: nx.DiGraph
    changepoints: list[int]
    interval_regime_labels: list[int]
    time_regime_labels: list[int]
    context_labels_by_target: dict[str, dict[int, int]]
    regime_labels_by_target: dict[str, dict[int, int]]
    variables: list[str]
    context_col: str
    tau_max: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkSample:
    df: pd.DataFrame
    true_summary_dag: nx.DiGraph
    spacetime: SpaceTimeSyntheticResult | None = None
    mixed: MixedSyntheticResult | None = None
    multi: MultiContextSyntheticResult | None = None
