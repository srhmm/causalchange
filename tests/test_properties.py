# tests/test_properties.py
import numpy as np
import networkx as nx
import pytest

from src.causalchange.causal_change import CausalChange
from src.causalchange.cc_types import DataMode, GraphSearch, GPType


@pytest.mark.parametrize("N", [2, 3, 4])  # N nodes
@pytest.mark.parametrize("graph_search", [GraphSearch.TOPIC])
def test_fit_produces_dag_and_valid_order_iid(N, graph_search):
    D = 100  # samples
    X = np.random.randn(D, N)  # (D, N)

    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=graph_search,
        score_type=GPType.EXACT,
        vb=0,
    )

    G = cc.fit(X)

    # Basic graph type & nodes
    assert isinstance(G, nx.DiGraph)
    assert set(G.nodes) == set(range(N))

    # DAG invariant
    assert nx.is_directed_acyclic_graph(G)

    # No self loops
    assert all(u != v for u, v in G.edges)

    # topological_order should be a permutation of nodes
    topo = cc.topological_order
    assert sorted(topo) == sorted(G.nodes)

    # topological_order should be consistent with edges
    pos = {node: i for i, node in enumerate(topo)}
    for u, v in G.edges:
        assert pos[u] < pos[v], f"Edge {u}->{v} violates topological order {topo}"
