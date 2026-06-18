from __future__ import annotations

import networkx as nx
import pandas as pd

from causalchange.core.results import GraphSearchResult, MultiContextResult, TabularResult
from causalchange.core.types import DataMode
from causalchange.engines.tabular import TabularDiscoveryEngine


class DummyDomain:
    def prepare_X(self, X):
        return X

    def nodes(self, X0):
        return list(X0.columns)

    def candidates(self, X0):
        return list(X0.columns)

    def allowed_edge(self, cause, effect):
        return cause != effect


class DummyContextPreproc:
    def make_contexts(self, X0):
        return {0: X0}

    def prepare_X(self, X):
        return X


class DummyScoring:
    @property
    def higher_is_better(self):
        return False

    def fit(self, X0):
        self.X0 = X0

    def local_score(self, df, effect, parents):
        return 0.0 if parents else 10.0

    def transition_gain(self, old_score, new_score):
        return old_score - new_score

    def gain_is_better(self, a, b):
        return a > b

    def raw_score_is_better(self, a, b):
        return a < b

    def score_significant(self, gain):
        return gain > 0


class DummyContextCombination:
    def combine_contexts(self, *, contexts, effect, parents, score_ctx):
        ctx, df = next(iter(contexts.items()))
        return MultiContextResult(total=score_ctx(df), diagnostics={"context": ctx})


class DummySearch:
    def run(self, *, nodes, candidates, allowed_edge, score_fun):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edge("x", "y")
        return GraphSearchResult(graph=graph, topological_order=["x", "y"], history=[{"ok": True}])


def test_tabular_engine_fit_local_score_and_discover():
    engine = TabularDiscoveryEngine(
        data_mode=DataMode.TABULAR,
        domain=DummyDomain(),
        context_preproc=DummyContextPreproc(),
        scoring=DummyScoring(),
        context_comb=DummyContextCombination(),
        search=DummySearch(),
    )

    engine.fit(pd.DataFrame({"x": [1, 2], "y": [3, 4]}))

    assert engine.local_score("y", ("x",)) == 0.0

    result = engine.discover()
    assert isinstance(result, TabularResult)
    assert list(result.graph.edges()) == [("x", "y")]
    assert result.history == [{"ok": True}]
