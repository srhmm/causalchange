
import pytest
import numpy as np
import networkx as nx

from causalchange.scoring.fit_cond_mixture import MixingType
from src.causalchange.cc_types import DataMode, GraphSearch, GPType
from src.causalchange.causal_change import CausalChange

from tests.utils.asserts import assert_is_dag, assert_topic_history_schema



@pytest.mark.slow
@pytest.mark.integration
def test_context_smoke():
    X = {0: np.random.default_rng(0).normal(size=(4, 30)),
         1: np.random.default_rng(0).normal(size=(4, 30))}
    cc = CausalChange(data_mode=DataMode.CONTEXTS, graph_search=GraphSearch.TOPIC, vb=0)
    G = cc.fit(X)
    assert set(G.nodes) == set(range(cc.N))


def _contexts_data(D=120, N=5, seed=0):
    rng = np.random.default_rng(seed)

    def gen_ctx(shift: float):
        e = rng.normal(size=(D, N))
        X = np.zeros((D, N), dtype=float)
        X[:, 0] = e[:, 0]
        X[:, 1] = (0.9 + shift) * X[:, 0] + e[:, 1]
        X[:, 2] = -0.5 * X[:, 0] + 0.4 * X[:, 1] + e[:, 2]
        X[:, 3] = (0.6 - shift) * X[:, 2] + e[:, 3]
        X[:, 4] = -0.3 * X[:, 1] + 0.6 * X[:, 3] + e[:, 4]
        return X

    return {0: gen_ctx(shift=0.0), 1: gen_ctx(shift=0.15)}


@pytest.mark.slow
@pytest.mark.integration
def test_contexts_topic_fit_produces_dag_and_history_schema():
    X = _contexts_data(D=80, N=5, seed=2)

    cc = CausalChange(
        data_mode=DataMode.CONTEXTS,
        graph_search=GraphSearch.TOPIC,
        score_type=GPType.EXACT,
        mixing_type=MixingType.SKIP,
        vb=0,
        extra_refinement=False,
    )

    G = cc.fit(X)

    assert isinstance(G, nx.DiGraph)
    assert set(G.nodes()) == set(range(next(iter(X.values())).shape[1]))
    assert cc.fitted_graph is True
    assert_is_dag(G)

    assert hasattr(cc, "topic_history")
    assert_topic_history_schema(cc.topic_history, N=cc.N)

