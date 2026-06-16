from __future__ import annotations

import networkx as nx

from causalchange.discovery.graph_tabular import GraphSearchTabularTopological


class DeterministicScoring:
    def transition_gain(self, old_score: float, new_score: float) -> float:
        return old_score - new_score

    def score_significant(self, gain: float) -> bool:
        return gain > 1.0

    def gain_is_better(self, a: float, b: float) -> bool:
        return a > b


def test_tabular_search_respects_allowed_edge():
    search = GraphSearchTabularTopological(scoring=DeterministicScoring())

    def score_fun(effect, parents):
        return 0.0 if parents else 10.0

    result = search.run(
        nodes=["x", "y"],
        candidates=["x", "y"],
        allowed_edge=lambda cause, effect: cause == "x" and effect == "y",
        score_fun=score_fun,
    )

    assert ("x", "y") in result.graph.edges()
    assert ("y", "x") not in result.graph.edges()


def test_tabular_search_returns_acyclic_graph():
    search = GraphSearchTabularTopological(scoring=DeterministicScoring())

    def score_fun(effect, parents):
        return 0.0 if parents else 10.0

    result = search.run(
        nodes=["x", "y", "z"],
        candidates=["x", "y", "z"],
        allowed_edge=lambda cause, effect: cause != effect,
        score_fun=score_fun,
    )

    assert nx.is_directed_acyclic_graph(result.graph)


def test_tabular_search_history_has_one_entry_per_candidate():
    search = GraphSearchTabularTopological(scoring=DeterministicScoring())

    def score_fun(effect, parents):
        return 0.0 if parents else 10.0

    result = search.run(
        nodes=["x", "y", "z"],
        candidates=["x", "y", "z"],
        allowed_edge=lambda cause, effect: cause != effect,
        score_fun=score_fun,
    )

    assert len(result.history) == 3
    assert all("source" in row for row in result.history)
    assert all("added_edges" in row for row in result.history)
    assert all("pruned_edges" in row for row in result.history)


def test_tabular_search_prunes_redundant_parent():
    search = GraphSearchTabularTopological(scoring=DeterministicScoring())

    def score_fun(effect, parents):
        parents = set(parents)

        if effect == "z" and "x" in parents:
            return 0.0
        if effect == "z" and "y" in parents:
            return 9.5
        return 10.0

    result = search.run(
        nodes=["x", "y", "z"],
        candidates=["x", "y", "z"],
        allowed_edge=lambda cause, effect: cause != effect,
        score_fun=score_fun,
    )

    assert nx.is_directed_acyclic_graph(result.graph)

    z_parents = set(result.graph.predecessors("z"))
    assert "x" in z_parents
    assert len(z_parents) == 1
