from __future__ import annotations

import networkx as nx

from causalchange.core.results import (
    CausalChangeResult,
    ChangepointResult,
    GraphSearchResult,
    PostProcessingResult,
    SCMClusteringResult,
    TemporalResult,
)


def test_causal_change_result_delegates_graph_properties():
    graph = nx.DiGraph()
    graph.add_edge("x", "y")
    graph_search = GraphSearchResult(graph=graph, topological_order=["x", "y"])
    postprocessing = PostProcessingResult(edge_strengths={("x", "y"): 1.5})

    result = CausalChangeResult(graph_search=graph_search, postprocessing=postprocessing)

    assert result.graph is graph
    assert result.topological_order == ["x", "y"]
    assert result.edge_strengths == {("x", "y"): 1.5}


def test_temporal_result_aliases_component_results():
    graph = nx.DiGraph()
    graph_search = GraphSearchResult(graph=graph)
    changepoint = ChangepointResult(changepoints=[10], changepoints_by_context={0: [10]})
    clusters = SCMClusteringResult(contexts={"x": {0: 0}}, regimes={"x": {0: 0}})

    result = TemporalResult(
        graph_search=graph_search,
        changepoint=changepoint,
        mechanism_clustering=clusters,
    )

    assert result.changepoints == [10]
    assert result.changepoints_by_context == {0: [10]}
    assert result.grid_clusters is clusters
