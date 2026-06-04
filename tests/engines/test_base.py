from __future__ import annotations

import networkx as nx
import pandas as pd

from causalchange.core.results import CausalChangeResult, GraphSearchResult
from causalchange.core.types import DataMode, PostprocessingMode
from causalchange.engines.base import BaseDiscoveryEngine


class DummyScoring:
    def transition_gain(self, old_score: float, new_score: float) -> float:
        return old_score - new_score


class DummyEngine(BaseDiscoveryEngine[object, DummyScoring, object]):
    def fit(self, X: pd.DataFrame):
        return self

    def local_score(self, effect, parents) -> float:
        return 0.0 if parents else 10.0

    def _run_discovery(self) -> CausalChangeResult:
        graph = nx.DiGraph()
        graph.add_edge("x", "y")
        return CausalChangeResult(graph_search=GraphSearchResult(graph=graph))


def test_base_engine_runs_optional_postprocessing():
    engine = DummyEngine(
        data_mode=DataMode.TABULAR,
        domain=object(),
        scoring=DummyScoring(),
        search=object(),
        postprocessing_mode=PostprocessingMode.EDGE_STRENGTHS,
    )

    result = engine.discover()

    assert result.edge_strengths == {("x", "y"): 10.0}
    assert result.postprocessing.diagnostics["edge_strengths"]["n_edges"] == 1
