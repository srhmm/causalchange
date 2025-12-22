import numpy as np
import pytest
import networkx as nx

from causalchange.old.causal_change_large import CausalChange
from causalchange._cc_types import DataMode, GraphSearch
from causalchange.old.scoring.fit_cond_mixture import MixingType


def test_defaults_are_set():
    cc = CausalChange()
    assert cc.data_mode == DataMode.IID
    assert cc.graph_search == GraphSearch.TOPIC
    assert cc.mixing_type == MixingType.SKIP
    assert cc.k_max == 5
    assert cc.lambda_mix == 1
    assert cc.vb == 0


def test_graph_search_data_mode_compatibility():
    CausalChange(data_mode=DataMode.IID,      graph_search=GraphSearch.TOPIC)
    CausalChange(data_mode=DataMode.IID,      graph_search=GraphSearch.GLOBE)
    CausalChange(data_mode=DataMode.CONTEXTS, graph_search=GraphSearch.GLOBE)
    CausalChange(data_mode=DataMode.CONTEXTS, graph_search=GraphSearch.TOPIC)
    CausalChange(data_mode=DataMode.CONTEXTS, graph_search=GraphSearch.CHAIN)
    CausalChange(data_mode=DataMode.CONTEXTS, graph_search=GraphSearch.COMBO)

    with pytest.raises(AssertionError):
        CausalChange(data_mode=DataMode.IID, graph_search=GraphSearch.CHAIN)

    with pytest.raises(AssertionError):
        CausalChange(data_mode=DataMode.IID, graph_search=GraphSearch.COMBO)

def test_mixed_requires_non_skip_mixing_type():
    with pytest.raises(AssertionError):
        CausalChange(data_mode=DataMode.MIXED, mixing_type=MixingType.SKIP)

    non_skip = next(m for m in MixingType if m != MixingType.SKIP)
    CausalChange(data_mode=DataMode.MIXED, mixing_type=non_skip)


def test_oracle_order_requires_true_graph_and_order():
    true_g = nx.DiGraph()
    true_g.add_nodes_from([0, 1])
    true_order = [0, 1]

    CausalChange(oracle_order=False, truths={"true_g": true_g})

    with pytest.raises(AssertionError):
        CausalChange(oracle_order=True, truths={"true_g": true_g})
    CausalChange(oracle_order=True, truths={"true_g": true_g, "true_order": true_order})


def test_init_and_check_x_array_iid():
    cc = CausalChange(data_mode=DataMode.IID)
    D, N = 100, 3  # D: samples, N: nodes
    X = np.random.randn(D, N)
    cc.init_and_check_X(X)

    assert cc.D == D
    assert cc.N == N
    assert cc.node_nms == ["0", "1", "2"]


def test_init_and_check_x_dict_contexts():
    cc = CausalChange(data_mode=DataMode.CONTEXTS)
    D, N = 50, 3
    X = {
        0: np.random.randn(D, N),
        1: np.random.randn(D, N),
    }
    cc.init_and_check_X(X)
    assert cc.D == D
    assert cc.N == N


def test_init_and_check_x_wrong_type_raises():
    cc = CausalChange(data_mode=DataMode.IID)
    with pytest.raises(AssertionError):
        cc.init_and_check_X({"not": "array"})


def test_node_names_length_mismatch():
    cc = CausalChange(data_mode=DataMode.IID, node_nms=["a", "b"])
    D, N = 10, 3
    X = np.random.randn(D, N)
    with pytest.raises(AssertionError):
        cc.init_and_check_X(X)
