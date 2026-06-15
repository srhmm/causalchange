from __future__ import annotations

import networkx as nx
import pandas as pd

from causalchange import (
    ChangepointMode,
    ChangepointScope,
    DataMode,
    GraphSearch,
    MechanismClusteringMethod,
    MechanismClusteringScope,
    ScoreType,
    StatisticalTestingMethod,
)
from causalchange.config.causal_change_config import CausalChangeConfigTemporal
from causalchange.core.results import (
    CausalChangeResult,
    ChangepointResult,
    GraphSearchResult,
    GridCell,
    PostProcessingResult,
    SCMClusteringResult,
    TemporalResult,
)
from causalchange.discovery.scm_clustering import TemporalSCMClustering
from causalchange.domain.temporal import TimeGrid


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

    changepoint = ChangepointResult(
        changepoints=[10],
        changepoints_by_context={0: [10]},
    )

    clusters = SCMClusteringResult(
        cell_clusters={
            "x": {
                GridCell(dataset_id=0, interval_id=0): 0,
                GridCell(dataset_id=0, interval_id=1): 1,
            }
        },
        intervals_by_context={
            0: [(0, 10), (10, 20)],
        },
    )

    result = TemporalResult(
        graph_search=graph_search,
        changepoint=changepoint,
        mechanism_clustering=clusters,
    )

    assert result.changepoints == [10]
    assert result.changepoints_by_context == {0: [10]}
    assert result.grid_clusters is clusters
    assert result.grid_clusters.cluster_for(target="x", dataset_id=0, interval_id=1) == 1


def test_temporal_scm_clustering_skip_returns_trivial_grid():
    cfg = CausalChangeConfigTemporal(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        clustering_scope=MechanismClusteringScope.SKIP,
        clustering_method=MechanismClusteringMethod.SKIP,
    )

    clustering = TemporalSCMClustering(cfg)

    panel = TimeGrid(
        datasets={0: pd.DataFrame({"x": [1, 2, 3, 4]})},
        variables=["x"],
    )

    result = clustering.fit_predict(panel=panel, changepoints=[])

    assert result.intervals_by_context == {0: [(0, 4)]}
    assert result.cell_clusters == {
        "x": {
            GridCell(dataset_id=0, interval_id=0): 0,
        }
    }
    assert result.diagnostics["mode"] == "skip"


def test_temporal_scm_clustering_supports_context_specific_intervals():
    cfg = CausalChangeConfigTemporal(
        data_mode=DataMode.TIME_CONTEXTS,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        changepoint_mode=ChangepointMode.ORACLE,
        changepoint_scope=ChangepointScope.PER_CONTEXT,
        fixed_changepoints=[2],
        clustering_scope=MechanismClusteringScope.SKIP,
        clustering_method=MechanismClusteringMethod.SKIP,
        testing_method=StatisticalTestingMethod.SKIP,
    )

    clustering = TemporalSCMClustering(cfg)

    panel = TimeGrid(
        datasets={
            "a": pd.DataFrame({"x": [1, 2, 3, 4, 5]}),
            "b": pd.DataFrame({"x": [1, 2, 3, 4, 5]}),
        },
        variables=["x"],
        context_col="context",
    )

    result = clustering.fit_predict(
        panel=panel,
        changepoints_by_context={
            "a": [2],
            "b": [3],
        },
    )

    assert result.intervals_by_context == {
        "a": [(0, 2), (2, 5)],
        "b": [(0, 3), (3, 5)],
    }

    assert result.cell_clusters["x"] == {
        GridCell(dataset_id="a", interval_id=0): 0,
        GridCell(dataset_id="a", interval_id=1): 0,
        GridCell(dataset_id="b", interval_id=0): 0,
        GridCell(dataset_id="b", interval_id=1): 0,
    }

    assert result.diagnostics["mode"] == "skip"
