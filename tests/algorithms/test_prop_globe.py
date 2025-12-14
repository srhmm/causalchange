# tests/test_prop_globe.py
import numpy as np
import networkx as nx

from hypothesis import given, settings, strategies as st

from causalchange.scoring.fit_cond_mixture import MixingType
from src.causalchange.cc_types import DataMode, GraphSearch, GPType
from src.causalchange.causal_change import CausalChange
from src.causalchange.dag.dag import DAG
from src.causalchange.util.upq import UPQ

from tests.utils.fake_edge_memoized_table import FakeEdgeMemoizedTable

"""Tests for GLOBE with fake scoring to test expected behavior"""

def make_cc_globe(N: int):
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


def patch_fake_scorer(cc: CausalChange, table: dict):
    # base/penalty provide a fallback for parent sets we don't explicitly table
    cc.edges_state = FakeEdgeMemoizedTable(
        cc.X, cc.data_mode, cc.score_type, cc.mixing_type,
        table=table,
        base=1000.0,
        penalty=10.0,
    )


@st.composite
def globe_score_tables(draw, N_min=3, N_max=5):
    """
    Build a random table that at least defines:
      score([]->j) and score([i]->j) for all i!=j
    Everything else can fall back to FakeEdgeMemoizedTable(base, penalty).
    """
    N = draw(st.integers(min_value=N_min, max_value=N_max))

    # Baseline empty-parent scores
    base_scores = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
            min_size=N,
            max_size=N
        )
    )

    table = {}
    for j in range(N):
        table[(j, frozenset())] = float(base_scores[j])

    # Singleton parent scores
    # allow them to be above/below base so gains can be +/-.
    for j in range(N):
        for i in range(N):
            if i == j:
                continue
            s_ij = draw(
                st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False)
            )
            table[(j, frozenset({i}))] = float(s_ij)

    return N, table


@settings(max_examples=200, deadline=None)
@given(globe_score_tables())
def test_globe_initial_edges_pops_max_gain_first(data):
    """
    Property:
      initial_edges uses priority = -gain*100
      => q.pop_task() returns the edge with the largest gain.
    """
    N, table = data
    cc = make_cc_globe(N)
    patch_fake_scorer(cc, table)

    dag = DAG(cc.X, cc.N, cc.edges_state)
    q = UPQ()
    q = dag.initial_edges(q, skip_insignificant=False)

    top = q.pop_task()

    # compute true argmax gain among all i->j
    best_edge = None
    best_gain = -float("inf")
    for j in range(N):
        s0 = table[(j, frozenset())]
        for i in range(N):
            if i == j:
                continue
            sij = table[(j, frozenset({i}))]
            gain = s0 - sij
            if gain > best_gain:
                best_gain = gain
                best_edge = (i, j)

    assert (top.i, top.j) == best_edge


@settings(max_examples=150, deadline=None)
@given(globe_score_tables(), st.integers(min_value=1, max_value=10))
def test_globe_forward_keeps_graph_acyclic_and_synced(data, n_steps):
    """
    Properties after running a few forward steps:
      - graph_state remains a DAG (no cycles)
      - dag_model adjacency and graph_state edges stay synced (your _assert_sync)
    """
    N, table = data
    cc = make_cc_globe(N)
    patch_fake_scorer(cc, table)

    dag = DAG(cc.X, cc.N, cc.edges_state)
    q = UPQ()
    q = dag.initial_edges(q, skip_insignificant=False)

    steps = 0
    while q.pq and steps < n_steps:
        try:
            q, dag = cc._graph_search_edgegreedy_forward_next(q, dag)
        except KeyError:
            break
        steps += 1

        # must remain acyclic
        assert nx.is_directed_acyclic_graph(cc.graph_state)

        # must remain synced
        cc._assert_sync(dag)
