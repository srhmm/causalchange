import numpy as np
import networkx as nx
import pytest

from causalchange.old.causal_change_large import CausalChange
from causalchange._cc_types import DataMode, GraphSearch, GPType



def _assert_graph_basic_invariants(G, N):
    assert isinstance(G, nx.DiGraph)
    assert set(G.nodes) == set(range(N))
    assert all(u != v for u, v in G.edges)
    assert nx.is_directed_acyclic_graph(G)


@pytest.mark.regression
def test_topic_is_deterministic_under_identical_inputs_and_seed():
    N, D = 4, 80
    rng = np.random.default_rng(0)
    X = rng.standard_normal((D, N))

    def run():
        cc = CausalChange(
            data_mode=DataMode.IID,
            graph_search=GraphSearch.TOPIC,
            score_type=GPType.EXACT,
            vb=0,
        )
        G = cc.fit(X)
        return cc, G

    cc1, G1 = run()
    cc2, G2 = run()

    _assert_graph_basic_invariants(G1, N)
    _assert_graph_basic_invariants(G2, N)

    assert list(cc1.topological_order) == list(cc2.topological_order)
    assert set(G1.edges) == set(G2.edges)


@pytest.mark.regression
def test_cycle_never_introduced_even_if_scores_would_encourage_it(monkeypatch):
    N, D = 3, 60
    rng = np.random.default_rng(1)
    X = rng.standard_normal((D, N))

    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.TOPIC,
        score_type=GPType.EXACT,
        vb=0,
    )

    original_score = getattr(cc, "_score")

    def biased_score(parents, child):
        p = set(parents)
        reward = 0.0
        if child == 1 and 0 in p:
            reward -= 100.0
        if child == 2 and 1 in p:
            reward -= 100.0
        if child == 0 and 2 in p:
            reward -= 100.0
        base, extra = 0.0, {}
        return base + reward

    monkeypatch.setattr(cc, "_score", biased_score)

    G = cc.fit(X)
    _assert_graph_basic_invariants(G, N)


    assert not (G.has_edge(0, 1) and G.has_edge(1, 2) and G.has_edge(2, 0))


@pytest.mark.regression
def test_topic_pruning_removes_edges_when_significance_rule_says_insignificant(monkeypatch):
    import causalchange.old.causal_change_large as cc_mod

    N, D = 4, 80
    rng = np.random.default_rng(2)
    X = rng.standard_normal((D, N))


    if hasattr(cc_mod, "is_insignificant"):
        monkeypatch.setattr(cc_mod, "is_insignificant", lambda *a, **k: True)
    else:
        pytest.skip("is_insignificant not found to monkeypatch; adjust module import path")

    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.TOPIC,
        score_type=GPType.EXACT,
        vb=0,
    )
    G = cc.fit(X)
    _assert_graph_basic_invariants(G, N)


    assert G.number_of_edges() <= N - 1


@pytest.mark.regression
def test_oracle_order_is_respected_even_if_scores_conflict():
    N, D = 4, 80
    rng = np.random.default_rng(3)
    X = rng.standard_normal((D, N))

    true_order = [3, 1, 0, 2]
    true_g = nx.DiGraph([(3, 1), (1, 0), (0, 2)])

    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.TOPIC,
        score_type=GPType.EXACT,
        oracle_order=True,
        truths={"true_order": true_order, "true_g": true_g},
        vb=0,
    )
    G = cc.fit(X)
    _assert_graph_basic_invariants(G, N)

    assert list(cc.topological_order) == true_order
