import numpy as np
import networkx as nx
import pytest

from src.causalchange.causal_change import CausalChange
from src.causalchange.cc_types import DataMode, GraphSearch, GPType

# If your suite already has TestableCausalChange/FakeEdgeMemoizedTable, import them:
# from tests.utils.test_causal_change import TestableCausalChange
# from tests.utils.fake_edges import FakeEdgeMemoizedTable


def _assert_graph_basic_invariants(G, N):
    assert isinstance(G, nx.DiGraph)
    assert set(G.nodes) == set(range(N))
    assert all(u != v for u, v in G.edges)
    assert nx.is_directed_acyclic_graph(G)


@pytest.mark.regression
def test_topic_is_deterministic_under_identical_inputs_and_seed():
    """
    Regression goal:
    - Catch accidental nondeterminism / iteration-order dependencies.
    - Doesn't assume a particular tie-break rule; just requires stability.
    """
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

    # Determinism criteria: same edge set + same topological order
    assert list(cc1.topological_order) == list(cc2.topological_order)
    assert set(G1.edges) == set(G2.edges)


@pytest.mark.regression
def test_cycle_never_introduced_even_if_scores_would_encourage_it(monkeypatch):
    """
    Regression goal:
    - If an edge would create a cycle, it must not be added.
    Strategy:
    - Force the scorer to *prefer* cycle-creating edges by monkeypatching the edge scorer.
    - Assert the final graph is still a DAG.
    """
    N, D = 3, 60
    rng = np.random.default_rng(1)
    X = rng.standard_normal((D, N))

    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.TOPIC,
        score_type=GPType.EXACT,
        vb=0,
    )

    # Patch cc._score to strongly "reward" adding parents that form a cycle.
    # This is a bit hacky but very effective as a regression guard.
    original_score = getattr(cc, "_score")

    def biased_score(parents, child):
        # Reward specifically a 0->1, 1->2, 2->0 cycle if possible
        p = set(parents)
        reward = 0.0
        if child == 1 and 0 in p:
            reward -= 100.0
        if child == 2 and 1 in p:
            reward -= 100.0
        if child == 0 and 2 in p:
            reward -= 100.0
        # Return (score, extra)
        base, extra = 0.0, {}
        return base + reward #, extra

    monkeypatch.setattr(cc, "_score", biased_score)

    G = cc.fit(X)
    _assert_graph_basic_invariants(G, N)

    # Stronger: explicitly assert the 3-cycle does NOT exist
    assert not (G.has_edge(0, 1) and G.has_edge(1, 2) and G.has_edge(2, 0))


@pytest.mark.regression
def test_topic_pruning_removes_edges_when_significance_rule_says_insignificant(monkeypatch):
    """
    Regression goal:
    - Ensure pruning step actually removes an edge when the significance test allows it.
    Notes:
    - This assumes your TOPIC refinement calls `is_insignificant(...)`.
    - We monkeypatch it to always return True, making pruning aggressive.
    """
    # Adjust import path to wherever is_insignificant is defined in your project.
    # Example (guess): src.causalchange.graph_utils or similar.
    # You'll need to change this to the correct module.
    import src.causalchange.causal_change as cc_mod

    N, D = 4, 80
    rng = np.random.default_rng(2)
    X = rng.standard_normal((D, N))

    # Aggressively prune everything deemed "insignificant"
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

    # With very aggressive pruning, we expect a sparse graph.
    # Avoid asserting exact edges; just assert "not dense".
    assert G.number_of_edges() <= N - 1


@pytest.mark.regression
def test_oracle_order_is_respected_even_if_scores_conflict():
    """
    Regression goal:
    - If oracle_order=True, the returned topological_order must equal truths["true_order"].
    """
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
