import numpy as np
import networkx as nx
import pytest

from causalchange.old.scoring.fit_cond_mixture import MixingType
from causalchange.old.causal_change_large import CausalChange
from causalchange._cc_types import DataMode, GraphSearch, GPType
from tests.utils.sample import sample_linear_sem, sample_time_series, sample_time_series_contexts, sample_linear_sem_mixed

"""End to end tests for all data types (DataMode) and algos (GraphSearch)"""


def _run_e2e_iid(graph_search: GraphSearch):
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    true_g = nx.from_numpy_array(adj, create_using=nx.DiGraph)

    X = sample_linear_sem(adj, n_samples=300, noise_std=0.2, seed=1)  # (D, N)

    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=graph_search,
        score_type=GPType.EXACT,
        truths={"true_g": true_g},
        vb=0,
    )

    G_hat = cc.fit(X)

    assert isinstance(G_hat, nx.DiGraph)
    assert set(G_hat.nodes) == set(true_g.nodes)
    assert nx.is_directed_acyclic_graph(G_hat)
    assert all(u != v for u, v in G_hat.edges)

    if graph_search.value in [GraphSearch.TOPIC.value, GraphSearch.CHAIN.value, GraphSearch.COMBO.value]:
        topo = cc.topological_order
        assert sorted(topo) == sorted(G_hat.nodes)

        pos = {node: i for i, node in enumerate(topo)}
        for u, v in G_hat.edges:
            assert pos[u] < pos[v], f"Edge {u}->{v} violates topological order {topo}"


def _run_e2e_contexts(graph_search: GraphSearch):
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    true_g = nx.from_numpy_array(adj, create_using=nx.DiGraph)

    X_ctx0 = sample_linear_sem(adj, n_samples=200, noise_std=0.2, seed=1)
    X_ctx1 = sample_linear_sem(adj, n_samples=250, noise_std=0.2, seed=2)

    X_contexts = {
        0: X_ctx0,
        1: X_ctx1,
    }

    cc = CausalChange(
        data_mode=DataMode.CONTEXTS,
        graph_search=graph_search,
        score_type=GPType.EXACT,
        truths={"true_g": true_g},
        vb=0,
    )

    G_hat = cc.fit(X_contexts)

    assert isinstance(G_hat, nx.DiGraph)
    assert set(G_hat.nodes) == set(true_g.nodes)
    assert nx.is_directed_acyclic_graph(G_hat)
    assert all(u != v for u, v in G_hat.edges)

    if graph_search.value in [GraphSearch.TOPIC.value, GraphSearch.CHAIN.value, GraphSearch.COMBO.value]:
        topo = cc.topological_order
        assert sorted(topo) == sorted(G_hat.nodes)

        pos = {node: i for i, node in enumerate(topo)}
        for u, v in G_hat.edges:
            assert pos[u] < pos[v], f"Edge {u}->{v} violates topological order {topo}"


def _run_e2e_mixed(graph_search: GraphSearch):
    pytest.importorskip("rpy2", reason="MIXED mode requires rpy2")
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    true_g = nx.from_numpy_array(adj, create_using=nx.DiGraph)

    X, mech_assign = sample_linear_sem_mixed(adj, n_samples=400, noise_std=0.2, seed=5)

    cc = CausalChange(
        data_mode=DataMode.MIXED,
        graph_search=graph_search,
        mixing_type=MixingType.MIX_LIN,
        truths={"true_g": true_g},
        vb=0,
    )

    G_hat = cc.fit(X)

    assert isinstance(G_hat, nx.DiGraph)
    assert set(G_hat.nodes) == set(true_g.nodes)
    assert nx.is_directed_acyclic_graph(G_hat)
    assert all(u != v for u, v in G_hat.edges)


    if graph_search.value in [GraphSearch.TOPIC.value]:
        topo = cc.topological_order
        assert sorted(topo) == sorted(G_hat.nodes)
        pos = {node: i for i, node in enumerate(topo)}
        for u, v in G_hat.edges:
            assert pos[u] < pos[v], f"Edge {u}->{v} violates topological order {topo}"


def _run_e2e_time(graph_search: GraphSearch):
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    true_g = nx.from_numpy_array(adj, create_using=nx.DiGraph)

    X_time = sample_time_series(adj, n_timepoints=300, noise_std=0.2, seed=3)  # (T, N)

    cc = CausalChange(
        data_mode=DataMode.TIME,
        graph_search=graph_search,
        score_type=GPType.EXACT,
        truths={"true_g": true_g},
        vb=0,
    )

    try:
        G_hat = cc.fit(X_time)
    except NotImplementedError:
        pytest.xfail("TIME graph search not implemented yet in CausalChange")
        return

    assert isinstance(G_hat, nx.DiGraph)
    assert set(G_hat.nodes) == set(true_g.nodes)
    assert nx.is_directed_acyclic_graph(G_hat)
    assert all(u != v for u, v in G_hat.edges)

    if graph_search.value in [GraphSearch.TOPIC.value]:
        topo = cc.topological_order
        assert sorted(topo) == sorted(G_hat.nodes)

        pos = {node: i for i, node in enumerate(topo)}
        for u, v in G_hat.edges:
            assert pos[u] < pos[v], f"Edge {u}->{v} violates topological order {topo}"


def _run_e2e_time_contexts(graph_search: GraphSearch):
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    true_g = nx.from_numpy_array(adj, create_using=nx.DiGraph)

    X_contexts = sample_time_series_contexts(
        adj_or_lagged=adj,
        n_contexts=2,
        n_timepoints_per_context=200,
        noise_std=0.2,
        seed=10,
    )

    cc = CausalChange(
        data_mode=DataMode.TIME_CONTEXTS,
        graph_search=graph_search,
        score_type=GPType.EXACT,
        truths={"true_g": true_g},
        vb=0,
    )

    try:
        G_hat = cc.fit(X_contexts)
    except NotImplementedError:
        pytest.xfail("TIME_CONTEXTS graph search not implemented yet in CausalChange")
        return

    assert isinstance(G_hat, nx.DiGraph)
    assert set(G_hat.nodes) == set(true_g.nodes)
    assert nx.is_directed_acyclic_graph(G_hat)
    assert all(u != v for u, v in G_hat.edges)

    if graph_search.value in [GraphSearch.TOPIC.value]:
        topo = cc.topological_order
        assert sorted(topo) == sorted(G_hat.nodes)

        pos = {node: i for i, node in enumerate(topo)}
        for u, v in G_hat.edges:
            assert pos[u] < pos[v], f"Edge {u}->{v} violates topological order {topo}"



@pytest.mark.parametrize(
    "data_mode",
    [DataMode.IID, DataMode.CONTEXTS, DataMode.MIXED, DataMode.TIME, DataMode.TIME_CONTEXTS],
)
@pytest.mark.parametrize(
    "graph_search",
    [GraphSearch.TOPIC, GraphSearch.GLOBE, GraphSearch.CHAIN],
)
def test_end_to_end(data_mode, graph_search):
    if not graph_search.is_compatible_with(data_mode):
        pytest.skip(f"{graph_search} not compatible with data_mode {data_mode}")

    if data_mode == DataMode.IID:
        _run_e2e_iid(graph_search)
    elif data_mode == DataMode.CONTEXTS:
        _run_e2e_contexts(graph_search)
    elif data_mode == DataMode.MIXED:
        pytest.importorskip("rpy2", reason="MIXED mode requires rpy2")
        _run_e2e_mixed(graph_search)
    elif data_mode == DataMode.TIME:
        _run_e2e_time(graph_search)
    elif data_mode == DataMode.TIME_CONTEXTS:
        _run_e2e_time_contexts(graph_search)
    else:
        pytest.skip(f"Data mode {data_mode} not handled in this test yet")
