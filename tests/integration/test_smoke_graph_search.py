from tests.utils.test_causal_change import TestableCausalChange


import numpy as np
import networkx as nx
import pytest

from src.causalchange.causal_change import CausalChange
from src.causalchange.cc_types import DataMode, GraphSearch, GPType


@pytest.mark.parametrize("N", [2, 3, 4])
@pytest.mark.parametrize("graph_search", [GraphSearch.TOPIC])
def test_fit_produces_dag_and_valid_order_iid(N, graph_search):
    D = 100
    X = np.random.randn(D, N)

    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=graph_search,
        score_type=GPType.EXACT,
        vb=0,
    )

    G = cc.fit(X)

    assert isinstance(G, nx.DiGraph)
    assert set(G.nodes) == set(range(N))
    assert nx.is_directed_acyclic_graph(G)

    assert all(u != v for u, v in G.edges)
    topo = cc.topological_order
    assert sorted(topo) == sorted(G.nodes)
    pos = {node: i for i, node in enumerate(topo)}
    for u, v in G.edges:
        assert pos[u] < pos[v], f"Edge {u}->{v} violates topological order {topo}"


def _check_graph_invariants(cc, G, N, expect_topo: bool):
    assert isinstance(G, nx.DiGraph)
    assert set(G.nodes) == set(range(N))
    assert all(u != v for u, v in G.edges)
    assert nx.is_directed_acyclic_graph(G)

    if expect_topo:
        topo = cc.topological_order
        assert sorted(topo) == sorted(G.nodes)

        pos = {node: i for i, node in enumerate(topo)}
        for u, v in G.edges:
            assert pos[u] < pos[v], f"Edge {u}->{v} violates topological order {topo}"


def _run_graph_search_iid(graph_search: GraphSearch, N: int = 3):
    D = 50
    X = np.random.randn(D, N)

    cc = TestableCausalChange(
        data_mode=DataMode.IID,
        graph_search=graph_search,
        score_type=GPType.EXACT,
        vb=0,
    )

    G = cc.fit(X)

    expect_topo = graph_search.value in [GraphSearch.TOPIC.value, GraphSearch.CHAIN.value]
    _check_graph_invariants(cc, G, N, expect_topo)


def _run_graph_search_contexts(graph_search: GraphSearch, N: int = 3):
    D0, D1 = 40, 45
    X = {
        0: np.random.randn(D0, N),
        1: np.random.randn(D1, N),
    }

    cc = TestableCausalChange(
        data_mode=DataMode.CONTEXTS,
        graph_search=graph_search,
        score_type=GPType.EXACT,
        vb=0,
    )

    G = cc.fit(X)

    expect_topo = graph_search.value in [GraphSearch.TOPIC.value, GraphSearch.CHAIN.value]
    _check_graph_invariants(cc, G, N, expect_topo)

from src.causalchange.scoring.fit_cond_mixture import MixingType

def _run_graph_search_mixed(graph_search: GraphSearch, N: int = 3):
    pytest.importorskip("rpy2", reason="MIXED mode requires rpy2")
    D = 60
    X = np.random.randn(D, N)

    cc = TestableCausalChange(
        data_mode=DataMode.MIXED,
        graph_search=graph_search,
        mixing_type=MixingType.MIX_LIN,
        score_type=GPType.EXACT,
        vb=0,
    )
    cc.init_and_check_X(X)
    cc.initialize()
    G = cc._graph_search()

    expect_topo = graph_search.value in [GraphSearch.TOPIC.value]
    _check_graph_invariants(cc, G, N, expect_topo)


def _run_graph_search_time(graph_search: GraphSearch, N: int = 3):
    T = 50
    X = np.random.randn(T, N)

    cc = TestableCausalChange(
        data_mode=DataMode.TIME,
        graph_search=graph_search,
        score_type=GPType.EXACT,
        vb=0,
    )
    try:
        G = cc.fit(X)
    except NotImplementedError:
        pytest.xfail("TIME graph search not implemented yet in CausalChange")
        return

    expect_topo = graph_search.value in [GraphSearch.TOPIC.value]
    _check_graph_invariants(cc, G, N, expect_topo)


def _run_graph_search_time_contexts(graph_search: GraphSearch, N: int = 3):
    T0, T1 = 40, 55
    X = {
        0: np.random.randn(T0, N),
        1: np.random.randn(T1, N),
    }

    cc = TestableCausalChange(
        data_mode=DataMode.TIME_CONTEXTS,
        graph_search=graph_search,
        score_type=GPType.EXACT,
        vb=0,
    )
    try:
        G = cc.fit(X)
    except NotImplementedError:
        pytest.xfail("TIME-CONTEXTS graph search not implemented yet in CausalChange")
        return
    expect_topo = graph_search.value in [GraphSearch.TOPIC.value]
    _check_graph_invariants(cc, G, N, expect_topo)


@pytest.mark.parametrize(
    "data_mode",
    [DataMode.IID, DataMode.CONTEXTS, DataMode.MIXED, DataMode.TIME, DataMode.TIME_CONTEXTS],
)
@pytest.mark.parametrize(
    "graph_search",
    [GraphSearch.TOPIC, GraphSearch.GLOBE, GraphSearch.CHAIN],
)


def test_graph_search(data_mode, graph_search):
    if not graph_search.is_compatible_with(data_mode):
        pytest.skip(f"{graph_search} is not compatible with data_mode {data_mode}")

    if data_mode == DataMode.IID:
        _run_graph_search_iid(graph_search, N=3)
    elif data_mode == DataMode.CONTEXTS:
        _run_graph_search_contexts(graph_search, N=3)
    elif data_mode == DataMode.MIXED:
        pytest.importorskip("rpy2", reason="MIXED mode requires rpy2")
        _run_graph_search_mixed(graph_search, N=3)
    elif data_mode == DataMode.TIME:
        _run_graph_search_time(graph_search, N=3)
    elif data_mode == DataMode.TIME_CONTEXTS:
        _run_graph_search_time_contexts(graph_search, N=3)
    else:
        pytest.skip(f"DataMode {data_mode} not handled in this test")


def test_topic_respects_oracle_order_when_provided():
    N = 3
    D = 50
    X = np.random.randn(D, N)

    true_order = [2, 0, 1]

    cc = TestableCausalChange(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.TOPIC,
        score_type=GPType.EXACT,
        oracle_order=True,
        truths={"true_order": true_order, "true_g": nx.DiGraph([(2, 0), (0, 1)])},
        vb=0,
    )

    G = cc.fit(X)
    assert nx.is_directed_acyclic_graph(G)

    topo = cc.topological_order
    assert topo == true_order, f"Expected oracle order {true_order}, got {topo}"
