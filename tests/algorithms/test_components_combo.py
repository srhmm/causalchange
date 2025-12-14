import numpy as np
import networkx as nx

from causalchange.scoring.fit_cond_mixture import MixingType
from src.causalchange.cc_types import DataMode, GraphSearch, GPType
from src.causalchange.causal_change import CausalChange
import src.causalchange.search.partition_search as ps


def make_cc_combo(N=4):
    X = {0: np.zeros((20, N), dtype=float), 1: np.zeros((20, N), dtype=float)}
    cc = CausalChange(
        data_mode=DataMode.CONTEXTS,
        graph_search=GraphSearch.COMBO,
        score_type=GPType.EXACT,
        mixing_type=MixingType.SKIP,
        vb=0,
    )
    cc.init_and_check_X(X)
    cc.graph_state = nx.DiGraph()
    cc.graph_state.add_nodes_from(range(N))
    return cc


def test_combo_next_picks_min_combined_score(monkeypatch):
    cc = make_cc_combo(N=4)
    cc.candidates = [0, 1, 2, 3]
    cc.topological_order = []

    d0 = {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0}
    d1 = {
        (0, 1): 0.0,
        (1, 0): 9.0,
        (2, 0): 9.5,
        (3, 0): 9.5,
    }
    dep = {0: 1.0, 1: 0.1, 2: 1.0, 3: 1.0}

    def fake_discrepancy_mmd(X_all, C_idx, i, parents):
        if len(parents) == 0:
            return d0[i]
        return d1.get((i, parents[0]), d0[i])

    def fake_resit_dep_score_joint_pairwise(X_all, C_idx, i, parents, reg):
        return dep[i], {}

    monkeypatch.setattr(ps, "discrepancy_mmd", fake_discrepancy_mmd)
    monkeypatch.setattr(ps, "resit_dep_score_joint_pairwise", fake_resit_dep_score_joint_pairwise)

    src = cc._graph_search_combo_next(
        lam_mix=0.5,
        X_all=None,
        C_idx=None,
        dep_reg=lambda X: X,
        indep_test_fun=lambda R, X: 1.0,
        K_max=1,
    )
    assert src == 1


def test_combo_add_edges_after_order_calls_builder(monkeypatch):
    cc = make_cc_combo(N=3)
    cc.topological_order = [0, 1, 2]
    cc.candidates = []

    called = {"ok": False}

    def fake_add_edges_combo_given_order(**kwargs):
        called["ok"] = True
        return [(0, 1), (1, 2)]

    monkeypatch.setattr(ps, "add_edges_combo_given_order", fake_add_edges_combo_given_order)

    cc._graph_search_combo_add_edges_after_order(lam_edge=0.5, gain_min=0.0)

    assert called["ok"]
    assert cc.graph_state.has_edge(0, 1)
    assert cc.graph_state.has_edge(1, 2)
