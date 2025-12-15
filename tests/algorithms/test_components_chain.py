import numpy as np
import networkx as nx

from src.causalchange.cc_types import DataMode, GraphSearch, GPType
from src.causalchange.causal_change import CausalChange


class FakeDiscrepancyTable:
    def __init__(self, table: dict, default: float = 0.0):
        self.table = table
        self.default = float(default)

    def discrepancy(self, child, parents):
        key = (int(child), frozenset(int(p) for p in parents))
        return float(self.table.get(key, self.default)), {}

    def score_edge(self, child, parents):
        return 0.0, {}


def make_cc_chain(N=4):
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
    cc.is_discrepancy_insignificant = lambda delta, eps=0.0: delta <= eps
    return cc


def patch_fake_discrepancy(cc, table, default=0.0):
    cc.edges_state = FakeDiscrepancyTable(table=table, default=default)


def test_chain_next_picks_node_with_smallest_discrepancy_reduction():
    cc = make_cc_chain(N=4)
    cc.candidates = [0, 1, 2, 3]
    cc.topological_order = []

    table = {}
    for j in range(4):
        table[(j, frozenset())] = 10.0

    table[(0, frozenset({1}))] = 0.0
    table[(1, frozenset({0}))] = 5.0
    table[(3, frozenset({0}))] = 7.0

    patch_fake_discrepancy(cc, table, default=10.0)

    src = cc._graph_search_chain_next(K_max=1, eps_add=0.0, allow_prev_as_parents=False)
    assert src == 2


def test_chain_prune_adds_edge_when_removal_increases_discrepancy():
    cc = make_cc_chain(N=3)
    cc.graph_state = nx.DiGraph()
    cc.graph_state.add_nodes_from(range(3))
    cc.topological_order = [0, 1, 2]

    table = {
        (2, frozenset({0, 1})): 0.0,
        (2, frozenset({0})): 10.0,
        (2, frozenset({1})): 0.0,
    }
    patch_fake_discrepancy(cc, table, default=10.0)

    cc._graph_search_chain_prune()

    assert cc.graph_state.has_edge(1, 2)
    assert not cc.graph_state.has_edge(0, 2)
