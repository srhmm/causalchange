import numpy as np
import networkx as nx

from hypothesis import given, settings, strategies as st

from causalchange.scoring.fit_cond_mixture import MixingType
from src.causalchange.cc_types import DataMode, GraphSearch, GPType
from src.causalchange.causal_change import CausalChange
import src.causalchange.search.partition_search as ps


def make_cc_combo(N: int):
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


@st.composite
def combo_mmd_tables(draw, N_min=3, N_max=6):
    N = draw(st.integers(min_value=N_min, max_value=N_max))
    d0s = draw(
        st.lists(
            st.floats(min_value=0.1, max_value=5000.0, allow_nan=False, allow_infinity=False),
            min_size=N, max_size=N,
        )
    )
    table = {}
    for i in range(N):
        table[(i, frozenset())] = float(d0s[i])
        for j in range(N):
            if j == i:
                continue
            d1 = draw(st.floats(min_value=0.0, max_value=table[(i, frozenset())], allow_nan=False, allow_infinity=False))
            table[(i, frozenset({j}))] = float(d1)
    return N, table


def expected_combo_source_lam1(N: int, table: dict, candidates: list[int]):
    best = None
    best_s = float("inf")
    for i in candidates:
        d0 = table[(i, frozenset())]
        dmin = d0
        for j in candidates:
            if j == i:
                continue
            dmin = min(dmin, table[(i, frozenset({j}))])
        r = max(0.0, d0 - dmin) / d0
        if r < best_s:
            best_s = r
            best = i
    return best


@settings(max_examples=200, deadline=None)
@given(combo_mmd_tables())
def test_combo_next_lam1_matches_mmd_rule(data, monkeypatch):
    N, table = data
    cc = make_cc_combo(N)
    cc.candidates = list(range(N))
    cc.topological_order = []

    def fake_discrepancy_mmd(X_all, C_idx, i, parents):
        key = (int(i), frozenset(int(p) for p in parents))
        return float(table[key])

    def fake_resit_dep_score_joint_pairwise(X_all, C_idx, i, parents, reg):
        return 0.0, {}

    monkeypatch.setattr(ps, "discrepancy_mmd", fake_discrepancy_mmd)
    monkeypatch.setattr(ps, "resit_dep_score_joint_pairwise", fake_resit_dep_score_joint_pairwise)

    src = cc._graph_search_combo_next(
        lam_mix=1.0,
        X_all=None,
        C_idx=None,
        dep_reg=lambda X: X,
        indep_test_fun=lambda R, X: 1.0,
        K_max=1,
    )
    expected = expected_combo_source_lam1(N, table, cc.candidates)
    assert src == expected


@settings(max_examples=100, deadline=None)
@given(combo_mmd_tables())
def test_combo_smoke_graph_is_dag_after_patched_build(data, monkeypatch):
    N, table = data
    cc = make_cc_combo(N)
    cc.candidates = list(range(N))
    cc.topological_order = []

    def fake_discrepancy_mmd(X_all, C_idx, i, parents):
        key = (int(i), frozenset(int(p) for p in parents))
        return float(table[key])

    def fake_resit_dep_score_joint_pairwise(X_all, C_idx, i, parents, reg):
        return 0.0, {}

    def fake_add_edges_combo_given_order(**kwargs):
        order = kwargs["topological_order"]
        edges = []
        for a, b in zip(order[:-1], order[1:]):
            edges.append((a, b))
        return edges

    monkeypatch.setattr(ps, "discrepancy_mmd", fake_discrepancy_mmd)
    monkeypatch.setattr(ps, "resit_dep_score_joint_pairwise", fake_resit_dep_score_joint_pairwise)
    monkeypatch.setattr(ps, "add_edges_combo_given_order", fake_add_edges_combo_given_order)

    cc._graph_search_combo_ordering(lam_mmd_hsic=0.5, lam_mix=1.0, K_max=1, vb=-10)

    assert sorted(cc.topological_order) == list(range(N))
    assert nx.is_directed_acyclic_graph(cc.graph_state)
