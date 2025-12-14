# tests/test_components_globe.py
import numpy as np
import networkx as nx

from causalchange.scoring.fit_cond_mixture import MixingType
from src.causalchange.cc_types import DataMode, GraphSearch, GPType

from src.causalchange.causal_change import CausalChange
from src.causalchange.dag.dag import DAG
from src.causalchange.util.upq import UPQ
from tests.utils.fake_edge_memoized_table import FakeEdgeMemoizedTable

def make_cc_globe(N=4):
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
    # deterministic significance: only positive gains are “significant”
    cc.is_score_insignificant = lambda gain: gain <= 0.0
    return cc

def patch_fake_scorer(cc, table):
    cc.edges_state = FakeEdgeMemoizedTable(
        cc.X, cc.data_mode, cc.score_type, cc.mixing_type, table=table, base=1000.0, penalty=50.0
    )

def test_globe_initial_edges_pop_best_gain_first():
    """
    initial_edges inserts all i->j with priority = -gain*100.
    So popping should return the *largest gain* first (most negative priority).
    """
    cc = make_cc_globe(N=3)

    # baseline
    table = {
        (0, frozenset()): 1000.0,
        (1, frozenset()): 1000.0,
        (2, frozenset()): 1000.0,
        # Make 2->1 very good: score([2]->1)=800 => gain=200
        (1, frozenset({2})): 800.0,
        # Make 0->1 mildly good: score([0]->1)=950 => gain=50
        (1, frozenset({0})): 950.0,
        # Make 0->2 bad: score([0]->2)=1010 => gain=-10
        (2, frozenset({0})): 1010.0,
    }
    patch_fake_scorer(cc, table)

    dag = DAG(cc.X, cc.N, cc.edges_state)
    q = UPQ()
    q = dag.initial_edges(q, skip_insignificant=False)

    top = q.pop_task()
    assert (top.i, top.j) == (2, 1)  # biggest gain first


def test_globe_forward_adds_edge_and_updates_graph_state_sync():
    cc = make_cc_globe(N=3)

    # Force best edge = 2->1
    table = {
        (0, frozenset()): 1000.0,
        (1, frozenset()): 1000.0,
        (2, frozenset()): 1000.0,
        (1, frozenset({2})): 800.0,   # gain 200
    }
    patch_fake_scorer(cc, table)

    dag = DAG(cc.X, cc.N, cc.edges_state)
    q = UPQ()
    q = dag.initial_edges(q, skip_insignificant=False)

    # Need cc._assert_sync to exist (you have it)
    q, dag = cc._graph_search_edgegreedy_forward_next(q, dag)

    assert cc.graph_state.has_edge(2, 1)
    # dag adjacency also has it
    A = dag.get_adj()
    assert A[2][1] != 0


def test_globe_flip_legality_checked_before_removal(monkeypatch):
    """
    Construct a situation where flipping u->v to v->u would create a cycle.
    Ensure update_children does NOT remove the edge.
    """
    cc = make_cc_globe(N=3)

    table = {
        (0, frozenset()): 1000.0,
        (1, frozenset()): 1000.0,
        (2, frozenset()): 1000.0,
    }
    patch_fake_scorer(cc, table)

    dag = DAG(cc.X, cc.N, cc.edges_state)
    # Build a DAG: 0->1, 1->2, 0->2
    cc._dm_add_edge(dag, 0, 1, score=None, gain=1.0, vb=-10)
    cc._dm_add_edge(dag, 1, 2, score=None, gain=1.0, vb=-10)
    cc._dm_add_edge(dag, 0, 2, score=None, gain=1.0, vb=-10)

    q = UPQ()

    # Make flip gain positive by monkeypatching _dm_eval_edge_flip, but flip is illegal (cycle)
    monkeypatch.setattr(cc, "_dm_eval_edge_flip", lambda dag_model, u, v: 1.0)

    # Attempt to flip 0->2 (would create 2->0, which forms cycle 2->0->1->2)
    q, dag = cc._graph_search_edgegreedy_update_children(child=2, node=0, edge_q=q, dag_model=dag)

    assert cc.graph_state.has_edge(0, 2)  # edge still there
    assert not cc.graph_state.has_edge(2, 0)


def test_globe_backward_refine_selects_best_parent_subset():
    """
    If node has parents [0,1,2], backward refine should keep subset with best gain.
    """
    cc = make_cc_globe(N=4)

    # For node 3, we want best subset = {1} (lowest score)
    table = {
        (3, frozenset({0,1,2})): 900.0,  # old score
        (3, frozenset({1})): 700.0,      # new much better
        (3, frozenset({0,2})): 950.0,
        (3, frozenset({0})): 910.0,
        (3, frozenset({2})): 920.0,
        (3, frozenset()): 1000.0,
    }
    patch_fake_scorer(cc, table)

    dag = DAG(cc.X, cc.N, cc.edges_state)
    cc._dm_add_edge(dag, 0, 3, score=None, gain=1.0, vb=-10)
    cc._dm_add_edge(dag, 1, 3, score=None, gain=1.0, vb=-10)
    cc._dm_add_edge(dag, 2, 3, score=None, gain=1.0, vb=-10)

    q = UPQ()
    q, dag = cc._graph_search_edgegreedy_backward_refine(node=3, edge_q=q, dag_model=dag)

    # Expect only 1->3 remains
    assert cc.graph_state.has_edge(1, 3)
    assert not cc.graph_state.has_edge(0, 3)
    assert not cc.graph_state.has_edge(2, 3)


def test_globe_initial_edges_skip_insignificant_filters_queue():
    """
    With skip_insignificant=True, edges with non-positive gain
    must NOT be inserted into the queue.
    """
    cc = make_cc_globe(N=3)

    table = {
        # baselines
        (0, frozenset()): 1000.0,
        (1, frozenset()): 1000.0,
        (2, frozenset()): 1000.0,

        # good edge: gain = 100
        (1, frozenset({2})): 900.0,

        # bad edge: gain = -10
        (1, frozenset({0})): 1010.0,
    }
    patch_fake_scorer(cc, table)

    dag = DAG(cc.X, cc.N, cc.edges_state)
    q = UPQ()
    q = dag.initial_edges(q, skip_insignificant=True)

    # Only the good edge (2 -> 1) should be present
    tasks = [q.pop_task() for _ in range(len(q.pq))]
    edges = {(t.i, t.j) for t in tasks}

    assert edges == {(2, 1)}


def test_globe_forward_skips_anticausal_edge():
    """
    If j->i already exists, i->j must NOT be added (anticausal protection).
    """
    cc = make_cc_globe(N=2)

    table = {
        (0, frozenset()): 1000.0,
        (1, frozenset()): 1000.0,

        # would be a good edge if allowed
        (1, frozenset({0})): 800.0,  # gain 200
    }
    patch_fake_scorer(cc, table)

    dag = DAG(cc.X, cc.N, cc.edges_state)

    # Pre-existing reverse edge: 1 -> 0
    cc._dm_add_edge(dag, 1, 0, score=None, gain=1.0, vb=-10)

    q = UPQ()
    q = dag.initial_edges(q, skip_insignificant=False)

    # Attempt forward step
    q, dag = cc._graph_search_edgegreedy_forward_next(q, dag)

    # Edge 0->1 must NOT be added
    assert not cc.graph_state.has_edge(0, 1)
    assert cc.graph_state.has_edge(1, 0)
