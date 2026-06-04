from __future__ import annotations

import networkx as nx

from causalchange.posthoc.temporal import fixed_partition_for_graph


def test_fixed_partition_for_graph_creates_default_temporal_partitions():
    graph = nx.DiGraph()
    graph.add_edge(("x", 1), ("x", 0))

    result = fixed_partition_for_graph(
        graph=graph,
        dataset_ids=["a", "b"],
        n_intervals=2,
    )

    from causalchange.core.results import GridCell

    assert result.intervals_by_context == {
        "a": [(0, 1), (1, 2)],
        "b": [(0, 1), (1, 2)],
    }
    assert result.cell_clusters == {
        "x": {
            GridCell(dataset_id="a", interval_id=0): 0,
            GridCell(dataset_id="a", interval_id=1): 1,
            GridCell(dataset_id="b", interval_id=0): 0,
            GridCell(dataset_id="b", interval_id=1): 1,
        }
    }
    assert result.diagnostics["mode"] == "fixed_posthoc"
