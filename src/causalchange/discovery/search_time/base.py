from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import networkx as nx
import numpy as np
import pandas as pd

Node = tuple[str, int]


@dataclass(frozen=True)
class SpaceTimePartitions:
    """
    contexts[target][dataset_id] = context_cluster_id
    regimes[target][regime_id] = regime_cluster_id
    """

    contexts: dict[str, dict[Any, int]]
    regimes: dict[str, dict[int, int]]
    diagnostics: dict[str, Any]


@dataclass
class SpaceTimeResult:
    graph: Any
    changepoints: list[int]
    partitions: SpaceTimePartitions
    topological_order: list[str] | None = None

    # For ChangepointScope.PER_CONTEXT, changepoints is the union of changepoints_by_context
    changepoints_by_context: dict[Any, list[int]] | None = None

    changepoint_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimePanel:
    """
    Collection of one or more aligned time-series datasets.

    For DataMode.TIME:
        datasets = {0: X}

    For DataMode.TIME_CONTEXTS:
        datasets = {context_id: X_context_without_context_col}
    """

    datasets: dict[Any, pd.DataFrame]
    variables: list[str]
    context_col: str | None = None

    @property
    def dataset_ids(self) -> list[Any]:
        return list(self.datasets.keys())

    @property
    def n_contexts(self) -> int:
        return len(self.datasets)

    def first_dataset(self) -> pd.DataFrame:
        return self.datasets[self.dataset_ids[0]]


# todo consider removing this
class SpaceTimeScoring(Protocol):
    tau_max: int

    def fit(self, X: pd.DataFrame) -> None: ...

    def set_time_windows(
        self,
        *,
        n_raw_samples: int,
        changepoints: list[int],
    ) -> None: ...

    def score_edge(
        self,
        X: pd.DataFrame,
        effect: Any,
        parents: tuple[Any, ...],
    ) -> float: ...

    def residual_signal(
        self,
        X: pd.DataFrame,
        *,
        graph: nx.DiGraph | None,
        variables: list[str],
    ) -> np.ndarray: ...

    def fit_panel(self, panel: TimePanel) -> None: ...

    def residual_signal_panel(
        self,
        panel: TimePanel,
        *,
        graph: nx.DiGraph | None,
        variables: list[str],
    ) -> np.ndarray: ...

    def score_edge_panel(
        self,
        *,
        panel: TimePanel,
        effect: Any,
        parents: tuple[Any, ...],
        partitions: SpaceTimePartitions,
    ) -> float: ...

    def transition_gain(self, old_score: float, new_score: float) -> float: ...

    def score_significant(self, gain: float) -> bool: ...

    def score_is_better(self, a: float, b: float) -> bool: ...
