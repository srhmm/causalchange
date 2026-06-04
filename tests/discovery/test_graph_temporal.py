from __future__ import annotations

from causalchange.discovery.graph_temporal import GraphSearchTemporalGreedy


class DummyScoring:
    def transition_gain(self, old_score: float, new_score: float) -> float:
        return old_score - new_score

    def score_significant(self, gain: float) -> bool:
        return False

    def gain_is_better(self, a: float, b: float) -> bool:
        return a > b


def test_temporal_greedy_search_builds_window_nodes():
    search = GraphSearchTemporalGreedy(scoring=DummyScoring())

    result = search.run(
        variables=["x", "y"],
        tau_max=1,
        allowed_edge=lambda cause, effect: effect[1] == 0 and cause != effect,
        score_fun=lambda effect, parents: 0.0,
    )

    assert ("x", 0) in result.graph.nodes
    assert ("x", 1) in result.graph.nodes
    assert result.graph.number_of_edges() == 0
    assert result.topological_order is not None
