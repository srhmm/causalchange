import numpy as np
import networkx as nx

from causalchange.scoring.fit_cond_mixture import MixingType
from src.causalchange.cc_types import DataMode, GraphSearch, GPType
from src.causalchange.causal_change import CausalChange



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


import src.causalchange.causal_change as cc_mod

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

    def fake_discrepancy_mmd(i, parents, X_all, C_idx, krr_lam, krr_sigma, mmd_sigma):
        i = int(i)
        if len(parents) == 0:
            return float(d0[i])
        return float(d1.get((i, int(parents[0])), d0[i]))

    def fake_resit_dep_score_joint_pairwise(effect, candidates, X_all, C_idx, **kwargs):
        return float(dep[int(effect)])

    monkeypatch.setattr(cc_mod, "discrepancy_mmd", fake_discrepancy_mmd)
    monkeypatch.setattr(cc_mod, "resit_dep_score_joint_pairwise", fake_resit_dep_score_joint_pairwise)

    src = cc._graph_search_combo_next(cc.candidates, lam_mix=0.5, krr_lam=1e-2)
    assert src == 1

def test_combo_add_edges_after_order_calls_builder(monkeypatch):
    cc = make_cc_combo(N=3)
    cc.topological_order = [0, 1, 2]
    cc.candidates = []

    called = {"ok": False}

    def fake_add_edges_combo_given_order(order, X_all, C_idx, **kwargs):
        called["ok"] = True
        G = nx.DiGraph()
        G.add_nodes_from(order)
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        added = [(0, 1, 1.0, 0.0, 0.0, 1.0, 0.5), (1, 2, 1.0, 0.0, 0.0, 1.0, 0.5)]
        return G, added

    monkeypatch.setattr(cc_mod, "add_edges_combo_given_order", fake_add_edges_combo_given_order)

    cc._graph_search_combo_add_edges_after_order(lam_edge=0.5, gain_min=0.0)

    assert called["ok"]
    assert set(cc.graph_state.edges()) == {(0, 1), (1, 2)}
