from __future__ import annotations

import networkx as nx
import pandas as pd

from causalchange.core.results import GraphSearchResult, MechanismClusteringResult, TemporalResult
from causalchange.core.types import ChangepointMode, ChangepointScope, DataMode, MechanismClusteringScope
from causalchange.domain.temporal import TemporalDomain
from causalchange.engines.temporal import TemporalDiscoveryEngine


class DummyTemporalScoring:
    @property
    def higher_is_better(self):
        return False

    def fit_panel(self, panel):
        self.panel = panel

    def set_time_windows(self, *, n_raw_samples, changepoints):
        self.windows = (n_raw_samples, tuple(changepoints))

    def local_score(self, X, effect, parents, *, ret_full_result=False, ret_residuals=False):
        return 0.0 if parents else 10.0

    def local_score_grid(self, *, panel, effect, parents, partitions):
        return 0.0 if parents else 10.0

    def transition_gain(self, old_score, new_score):
        return old_score - new_score

    def gain_is_better(self, a, b):
        return a > b

    def raw_score_is_better(self, a, b):
        return a < b

    def score_significant(self, gain):
        return gain > 0


class DummyChangepointDetection:
    changepoints_by_context_ = None
    diagnostics_ = {"mode": "none"}

    def detect(self, X=None, *, time_grid=None, graph=None, scorer=None, variables=None):
        return []


class DummyClustering:
    def fit_predict(self, X=None, *, panel=None, graph=None, changepoints=None):
        return MechanismClusteringResult(
            contexts={"x": {0: 0}},
            regimes={"x": {0: 0}},
            diagnostics={"mode": "dummy"},
        )


class DummyTemporalSearch:
    def run(self, *, variables, tau_max, allowed_edge, score_fun):
        graph = nx.DiGraph()
        graph.add_edge(("x", 1), ("x", 0))
        return GraphSearchResult(graph=graph, topological_order=["x"], history=[{"ok": True}])


def test_temporal_engine_fit_and_discover_with_dummy_components():
    engine = TemporalDiscoveryEngine(
        data_mode=DataMode.TIME,
        domain=TemporalDomain(tau_max=1),
        scoring=DummyTemporalScoring(),
        search=DummyTemporalSearch(),
        changepoint_detection=DummyChangepointDetection(),
        scm_clustering=DummyClustering(),
        clustering_scope=MechanismClusteringScope.SKIP,
        context_col="context",
        tau_max=1,
        changepoint_mode=ChangepointMode.SKIP,
        changepoint_scope=ChangepointScope.SKIP,
        max_iter=1,
    )

    engine.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    result = engine.discover()

    assert isinstance(result, TemporalResult)
    assert result.changepoints == []
    assert list(result.graph.edges()) == [(("x", 1), ("x", 0))]
