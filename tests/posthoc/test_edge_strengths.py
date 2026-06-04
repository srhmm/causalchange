from __future__ import annotations

import networkx as nx

from causalchange.posthoc.edge_strengths import compute_edge_strengths, compute_postprocessing_result


def test_compute_edge_strengths_uses_local_score_and_transition_gain():
    graph = nx.DiGraph()
    graph.add_edge("x", "y")

    def local_score(effect, parents):
        return 0.0 if parents else 10.0

    strengths = compute_edge_strengths(
        graph,
        local_score=local_score,
        transition_gain=lambda old, new: old - new,
    )

    assert strengths == {("x", "y"): 10.0}


def test_compute_postprocessing_result_wraps_edge_strengths():
    graph = nx.DiGraph()
    graph.add_edge("x", "y")

    result = compute_postprocessing_result(
        graph,
        local_score=lambda effect, parents: 0.0 if parents else 5.0,
        transition_gain=lambda old, new: old - new,
    )

    assert result.edge_strengths == {("x", "y"): 5.0}
    assert result.diagnostics["edge_strengths"]["n_edges"] == 1
