from __future__ import annotations

from causalchange.discovery.graph_temporal import GraphSearchTemporalGreedy


class DummyScoring:
    def transition_gain(self, old_score: float, new_score: float) -> float:
        return old_score - new_score

    def score_significant(self, gain: float) -> bool:
        return gain > 1.0

    def gain_is_better(self, a: float, b: float) -> bool:
        return a > b


def test_temporal_graph_search_adds_expected_lagged_edge():
    search = GraphSearchTemporalGreedy(scoring=DummyScoring())

    def local_score(effect, parents):
        parents = set(parents)
        if effect == ("y", 0) and ("x", 1) in parents:
            return 0.0
        return 10.0

    result = search.run(
        variables=["x", "y"],
        tau_max=1,
        allowed_edge=lambda cause, effect: cause == ("x", 1) and effect == ("y", 0),
        score_fun=local_score,
    )

    assert (("x", 1), ("y", 0)) in result.graph.edges
    assert (("y", 1), ("x", 0)) not in result.graph.edges
    assert ("x", 0) in result.graph.nodes
    assert ("y", 1) in result.graph.nodes
