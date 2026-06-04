from __future__ import annotations

from causalchange.discovery.graph_tabular import GraphSearchTabularTopological


class DummyScoring:
    def transition_gain(self, old_score: float, new_score: float) -> float:
        return old_score - new_score

    def score_significant(self, gain: float) -> bool:
        return gain > 0.1

    def gain_is_better(self, a: float, b: float) -> bool:
        return a > b


def test_tabular_topological_search_returns_graph_result():
    search = GraphSearchTabularTopological(scoring=DummyScoring())

    def score_fun(effect, parents):
        if effect == "y" and "x" in parents:
            return 0.0
        return 10.0

    result = search.run(
        nodes=["x", "y"],
        candidates=["x", "y"],
        allowed_edge=lambda cause, effect: cause != effect,
        score_fun=score_fun,
    )

    assert set(result.graph.nodes) == {"x", "y"}
    assert result.topological_order is not None
    assert len(result.history) == 2
