import networkx as nx
import numpy as np


def assert_is_dag(G: nx.DiGraph):
    assert isinstance(G, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(G)


def assert_same_edge_support_from_adj(G: nx.DiGraph, A: np.ndarray):
    A_nx = nx.to_numpy_array(G, dtype=int)
    assert A.shape == A_nx.shape
    assert np.all((A != 0) == (A_nx != 0))


def assert_topic_history_schema(topic_history: list, N: int):
    assert isinstance(topic_history, list)
    assert len(topic_history) == N
    required = {
        "iteration",
        "source",
        "topological_order",
        "remaining_candidates",
        "source_selection",
        "added_edges",
        "pruned_edges",
        "outgoing_scores",
        "incoming_scores",
    }
    for it, rec in enumerate(topic_history, start=1):
        assert required.issubset(rec.keys())
        assert rec["iteration"] == it
        assert len(rec["topological_order"]) == it
