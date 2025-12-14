import numpy as np
import networkx as nx
import pytest

from causalchange.scoring.fit_cond_mixture import MixingType
from src.causalchange.cc_types import DataMode, GraphSearch, GPType
from src.causalchange.causal_change import CausalChange
from src.causalchange.dag.dag import DAG
from src.causalchange.util.upq import UPQ
from tests.utils.fake_edge_memoized_table import FakeEdgeMemoizedTable



def make_cc_globe(N=3):
    X = np.zeros((20, N), dtype=float)
    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.GLOBE,
        score_type=GPType.EXACT,
        mixing_type=MixingType.SKIP,
        vb=0,
    )
    cc.init_and_check_X(X)
    cc.graph_state = nx.DiGraph()
    cc.graph_state.add_nodes_from(range(N))
    cc.is_score_insignificant = lambda gain: gain <= 0.0
    return cc


def patch_fake_scorer(cc, table):
    cc.edges_state = FakeEdgeMemoizedTable(
        cc.X, cc.data_mode, cc.score_type, cc.mixing_type, table=table, base=1000.0, penalty=50.0
    )


def test_globe_update_children_checks_flip_legality_before_removal(monkeypatch):
    cc = make_cc_globe(N=3)

    table = {
        (0, frozenset()): 1000.0,
        (1, frozenset()): 1000.0,
        (2, frozenset()): 1000.0,
    }
    patch_fake_scorer(cc, table)

    dag = DAG(cc.X, cc.N, cc.edges_state)

    cc._dm_add_edge(dag, 0, 1, score=None, gain=1.0, vb=-10)
    cc._dm_add_edge(dag, 1, 2, score=None, gain=1.0, vb=-10)
    cc._dm_add_edge(dag, 0, 2, score=None, gain=1.0, vb=-10)

    monkeypatch.setattr(cc, "_dm_eval_edge_flip", lambda dag_model, u, v: 1.0)

    q = UPQ()
    q, dag = cc._graph_search_edgegreedy_update_children(child=2, node=0, edge_q=q, dag_model=dag)

    assert cc.graph_state.has_edge(0, 2)
    assert not cc.graph_state.has_edge(2, 0)
