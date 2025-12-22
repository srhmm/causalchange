import numpy as np
import networkx as nx

from hypothesis import given, settings, strategies as st

from causalchange._cc_types import DataMode, GraphSearch, GPType
from causalchange.old.causal_change_large import CausalChange


class FakeDiscrepancyTable:
    def __init__(self, table: dict, default: float = 0.0):
        self.table = table
        self.default = float(default)

    def discrepancy(self, child, parents):
        key = (int(child), frozenset(int(p) for p in parents))
        return float(self.table.get(key, self.default)), {}

    def score_edge(self, child, parents):
        return 0.0, {}


def make_cc_chain(N: int):
    X = {0: np.zeros((20, N), dtype=float), 1: np.zeros((20, N), dtype=float)}
    cc = CausalChange(
        data_mode=DataMode.CONTEXTS,
        graph_search=GraphSearch.CHAIN,
        score_type=GPType.EXACT,
        vb=0,
    )
    cc.init_and_check_X(X)
    cc.graph_state = nx.DiGraph()
    cc.graph_state.add_nodes_from(range(N))
    return cc


def patch_fake_discrepancy(cc: CausalChange, table: dict, default: float = 0.0):
    cc.edges_state = FakeDiscrepancyTable(table=table, default=default)


@st.composite
def chain_discrepancy_tables(draw, N_min=3, N_max=6):
    N = draw(st.integers(min_value=N_min, max_value=N_max))
    table = {}
    d0s = draw(
        st.lists(
            st.floats(min_value=0.1, max_value=5000.0, allow_nan=False, allow_infinity=False),
            min_size=N, max_size=N,
        )
    )
    for j in range(N):
        d0 = float(d0s[j])
        table[(j, frozenset())] = d0
        for i in range(N):
            if i == j:
                continue
            d1 = draw(
                st.floats(min_value=0.0, max_value=d0, allow_nan=False, allow_infinity=False)
            )
            table[(j, frozenset({i}))] = float(d1)
    return N, table


def expected_chain_source_k1(N: int, table: dict, candidates: list[int]):
    best = None
    best_s = float("inf")
    for j in candidates:
        d0 = table[(j, frozenset())]
        dmin = d0
        for i in candidates:
            if i == j:
                continue
            dmin = min(dmin, table[(j, frozenset({i}))])
        s = d0 - dmin
        if s < best_s:
            best_s = s
            best = j
    return best


@settings(max_examples=200, deadline=None)
@given(chain_discrepancy_tables())
def test_chain_next_k1_matches_min_reduction_rule(data):
    N, table = data
    cc = make_cc_chain(N)
    patch_fake_discrepancy(cc, table)

    cc.candidates = list(range(N))
    cc.topological_order = []

    src = cc._graph_search_chain_next(K_max=1, eps_add=0.0, allow_prev_as_parents=False)
    expected = expected_chain_source_k1(N, table, cc.candidates)
    assert src == expected


@settings(max_examples=100, deadline=None)
@given(chain_discrepancy_tables())
def test_chain_full_order_has_permutation_and_dag(data):
    N, table = data
    cc = make_cc_chain(N)
    patch_fake_discrepancy(cc, table, default=max(table.values()))

    cc._graph_search_chain()

    assert sorted(cc.topological_order) == list(range(N))
    assert nx.is_directed_acyclic_graph(cc.graph_state)
    assert set(cc.graph_state.nodes()) == set(range(N))
