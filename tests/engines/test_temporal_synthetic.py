from __future__ import annotations

from typing import Any

import pandas as pd

from causalchange.core.results import SCMClusteringResult, TemporalResult
from causalchange.core.types import (
    ChangepointMode,
    ChangepointScope,
    DataMode,
    MechanismClusteringScope,
    PostprocessingMode,
)
from causalchange.discovery.graph_temporal import GraphSearchTemporalGreedy
from causalchange.domain.temporal import TemporalDomain
from causalchange.engines.temporal import TemporalDiscoveryEngine


class SyntheticTemporalScoring:
    """Deterministic temporal local scores for x(t-1) -> y(t)."""

    @property
    def higher_is_better(self) -> bool:
        return False

    def fit_panel(self, panel) -> None:
        self.panel = panel

    def set_time_windows(self, *, n_raw_samples: int, changepoints: list[int]) -> None:
        self.n_raw_samples = n_raw_samples
        self.changepoints = list(changepoints)

    def local_score(self, X: pd.DataFrame, effect: tuple[str, int], parents, **kwargs) -> float:
        parents = set(parents)
        if effect == ("y", 0) and ("x", 1) in parents:
            return 0.0
        return 10.0

    def local_score_grid(self, *, panel, effect, parents, partitions) -> float:
        return self.local_score(panel.first_dataset(), effect, parents)

    def transition_gain(self, old_score: float, new_score: float) -> float:
        return old_score - new_score

    def gain_is_better(self, a: float, b: float) -> bool:
        return a > b

    def raw_score_is_better(self, a: float, b: float) -> bool:
        return a < b

    def score_significant(self, gain: float) -> bool:
        return gain > 1.0


class NoChangepoints:
    changepoints_by_context_: dict[Any, list[int]] | None = None
    diagnostics_: dict[str, Any] = {"mode": "skip"}

    def detect(self, *args, **kwargs) -> list[int]:
        return []


class NoMechanismClustering:
    def fit_predict(
        self,
        *,
        panel,
        graph,
        changepoints,
        changepoints_by_context=None,
        scorer=None,
    ):
        return SCMClusteringResult(
            cell_clusters={},
            intervals_by_context={},
            diagnostics={"mode": "skip"},
        )


def test_temporal_engine_synthetic_recovers_expected_lagged_edge_and_postprocessing():
    scoring = SyntheticTemporalScoring()

    engine = TemporalDiscoveryEngine(
        data_mode=DataMode.TIME,
        domain=TemporalDomain(tau_max=1),
        scoring=scoring,
        search=GraphSearchTemporalGreedy(scoring=scoring),
        changepoint_detection=NoChangepoints(),
        scm_clustering=NoMechanismClustering(),
        clustering_scope=MechanismClusteringScope.SKIP,
        context_col="context",
        tau_max=1,
        changepoint_mode=ChangepointMode.SKIP,
        changepoint_scope=ChangepointScope.SKIP,
        max_iter=1,
        postprocessing_mode=PostprocessingMode.EDGE_STRENGTHS,
    )

    X = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 0.0, 1.0, 2.0, 3.0],
        }
    )

    result = engine.fit(X).discover()

    assert isinstance(result, TemporalResult)
    assert (("x", 1), ("y", 0)) in result.graph.edges
    assert result.changepoints == []
    assert result.edge_strengths[(("x", 1), ("y", 0))] == 10.0
