import numpy as np
import networkx as nx

from causalchange.old.scoring.fit_cond_mixture import MixingType
from causalchange._cc_types import DataMode, GraphSearch, GPType
from causalchange.old.causal_change_large import CausalChange


def _make_true_g(N=4):
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from([(0, 1), (0, 2), (2, 3)])
    return G


def test_oracle_G_returns_true_graph_unchanged():
    N = 4
    X = np.zeros((20, N), dtype=float)
    true_g = _make_true_g(N)

    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.GLOBE,
        score_type=GPType.EXACT,
        mixing_type=MixingType.SKIP,
        truths={"true_g": true_g},
        oracle_G=True,
        vb=0,
    )

    G = cc.fit(X)

    assert set(G.edges()) == set(true_g.edges())
    assert set(true_g.edges()) == {(0, 1), (0, 2), (2, 3)}


def test_oracle_order_topic_uses_given_true_order_for_sources():
    N = 4
    X = {0: np.zeros((30, N), dtype=float), 1: np.zeros((30, N), dtype=float)}
    true_g = _make_true_g(N)
    true_order = [2, 0, 1, 3]

    cc = CausalChange(
        data_mode=DataMode.CONTEXTS,
        graph_search=GraphSearch.TOPIC,
        score_type=GPType.EXACT,
        mixing_type=MixingType.SKIP,
        truths={"true_g": true_g, "true_order": true_order},
        oracle_order=True,
        vb=0,
        extra_refinement=False,
    )

    G = cc.fit(X)

    assert hasattr(cc, "topological_order")
    assert cc.topological_order == true_order
    assert set(G.nodes()) == set(range(N))
