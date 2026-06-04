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

    assert result.contexts == {"x": {"a": 0, "b": 0}}
    assert result.regimes == {"x": {0: 0, 1: 1}}
    assert result.diagnostics["mode"] == "fixed_posthoc"
