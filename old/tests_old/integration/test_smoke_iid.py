
import pytest
import numpy as np
import networkx as nx

from causalchange.old.scoring.fit_cond_mixture import MixingType
from causalchange._cc_types import DataMode, GraphSearch, GPType
from causalchange.old.causal_change_large import CausalChange

from tests.utils.asserts import assert_is_dag



@pytest.mark.slow
@pytest.mark.integration
def test_iid_smoke():
    X = np.random.default_rng(0).normal(size=(4, 30))
    cc = CausalChange(data_mode=DataMode.IID, graph_search=GraphSearch.TOPIC, vb=0)
    G = cc.fit(X)
    assert set(G.nodes) == set(range(cc.N))


def _linear_sem_iid(D=200, N=5, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((D, N), dtype=float)

    e = rng.normal(size=(D, N))
    X[:, 0] = e[:, 0]
    X[:, 1] = 0.8 * X[:, 0] + e[:, 1]
    X[:, 2] = -0.6 * X[:, 0] + 0.5 * X[:, 1] + e[:, 2]
    X[:, 3] = 0.7 * X[:, 2] + e[:, 3]
    X[:, 4] = -0.4 * X[:, 1] + 0.6 * X[:, 3] + e[:, 4]
    return X


@pytest.mark.slow
@pytest.mark.integration
def test_iid_fit_returns_dag_and_sets_flags():
    X = _linear_sem_iid(D=150, N=5, seed=1)

    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.GLOBE,
        score_type=GPType.EXACT,
        mixing_type=MixingType.SKIP,
        vb=0,
    )

    G = cc.fit(X)

    assert isinstance(G, nx.DiGraph)
    assert set(G.nodes()) == set(range(X.shape[1]))
    assert cc.fitted_graph is True
    assert_is_dag(G)
    assert len(list(nx.topological_sort(G))) == X.shape[1]