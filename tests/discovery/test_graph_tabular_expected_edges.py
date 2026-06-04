from __future__ import annotations

from causalchange.discovery.graph_tabular import GraphSearchTabularTopological


class DummyScoring:
    def transition_gain(self, old_score: float, new_score: float) -> float:
        return old_score - new_score

    def score_significant(self, gain: float) -> bool:
        return gain > 1.0

    def gain_is_better(self, a: float, b: float) -> bool:
        return a > b


def test_tabular_graph_search_adds_only_expected_edge():
    search = GraphSearchTabularTopological(scoring=DummyScoring())

    def local_score(effect, parents):
        parents = set(parents)
        if effect == "y" and "x" in parents:
            return 0.0
        return 10.0

    result = search.run(
        nodes=["x", "y", "z"],
        candidates=["x", "y", "z"],
        allowed_edge=lambda cause, effect: (cause, effect) == ("x", "y"),
        score_fun=local_score,
    )

    assert set(result.graph.nodes) == {"x", "y", "z"}
    assert set(result.graph.edges) == {("x", "y")}
