import numpy as np
import networkx as nx

from causalchange.scoring.fit_cond_mixture import MixingType
from src.causalchange.cc_types import DataMode, GraphSearch, GPType
from src.causalchange.causal_change import CausalChange
from tests.utils.fake_edge_memoized_table import FakeEdgeMemoizedTable


def make_cc_topic(N=4):
    X = np.zeros((20, N), dtype=float)
    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.TOPIC,
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


def test_topic_source_selection_incoming_pressure_prefers_true_source():

    cc = make_cc_topic(N=4)
    cc.graph_state = nx.DiGraph()
    cc.graph_state.add_nodes_from(range(cc.N))

    candidates = list(range(cc.N))

    table = {}
    for j in range(4):
        table[(j, frozenset())] = 1000.0 + j

    for j in range(4):
        for p in range(4):
            if p == j:
                continue
            table[(j, frozenset({p}))] = table[(j, frozenset())]

    table[(1, frozenset({0}))] = 900.0   # gain 101
    table[(3, frozenset({0}))] = 850.0   # gain 153
    table[(3, frozenset({1}))] = 860.0   # gain 143
    table[(0, frozenset({3}))] = 870.0   # gain 130

    for p in [0, 1, 3]:
        table[(2, frozenset({p}))] = table[(2, frozenset())] + 200.0  # gain = -200

    patch_fake_scorer(cc, table)

    source, meta = cc._graph_search_topological_next(candidates)
    assert source == 2
    assert "incoming_pressure" in meta
    assert meta["ranking"][0]["node"] == 2


def test_topic_add_outgoing_adds_only_significant_edges():
    cc = make_cc_topic(N=4)
    cc.graph_state.add_nodes_from(range(4))
    cc.candidates = [1, 2, 3]
    cc.topological_order = [0]

    table = {
        (1, frozenset()): 1000.0,
        (1, frozenset({0})): 900.0,   # gain +100
        (2, frozenset()): 1000.0,
        (2, frozenset({0})): 1005.0,  # gain -5
        (3, frozenset()): 1000.0,
        (3, frozenset({0})): 1000.0,  # gain 0
    }
    patch_fake_scorer(cc, table)

    added, scores = cc._graph_search_topological_add_outgoing(source=0)
    assert (0, 1) in cc.graph_state.edges
    assert (0, 2) not in cc.graph_state.edges
    assert (0, 3) not in cc.graph_state.edges
    assert len(added) == 1
    assert any(s["to"] == 1 and s["significant"] for s in scores)


def test_topic_remove_ingoing_prunes_edges_with_insignificant_harm():
    cc = make_cc_topic(N=3)
    cc.graph_state.add_nodes_from(range(3))

    cc.graph_state.add_edge(0, 2)
    cc.graph_state.add_edge(1, 2)

    # Make removing parent 0 produce insignificant harm (new_score - old_score <= 0),
    # but removing parent 1 causes harm (new_score - old_score > 0).
    #
    # old_score = score({0,1})->2, new_score = score({1})->2 or score({0})->2
    table = {
        (2, frozenset({0, 1})): 900.0,
        (2, frozenset({1})): 900.0,   # harm 0
        (2, frozenset({0})): 950.0,   # harm 50
        (2, frozenset()): 1000.0,
    }
    patch_fake_scorer(cc, table)

    pruned, incoming_scores = cc._graph_search_topological_remove_ingoing(source=2)

    assert (0, 2) not in cc.graph_state.edges
    assert (1, 2) in cc.graph_state.edges
    assert any(e["from"] == 0 and e["to"] == 2 for e in pruned)
    assert len(incoming_scores) > 0


def test_topic_history_contains_expected_fields():
    cc = make_cc_topic(N=3)
    cc.graph_state.add_nodes_from(range(3))

    table = {}
    for child in range(3):
        table[(child, frozenset())] = 1000.0 + child
        for p in range(3):
            if p == child:
                continue
            table[(child, frozenset({p}))] = 990.0 + child  # gain 10

    patch_fake_scorer(cc, table)

    cc._graph_search_topological()

    assert isinstance(cc.topic_history, list)
    assert len(cc.topic_history) == 3
    keys = set(cc.topic_history[0].keys())
    expected = {
        "iteration", "source", "topological_order", "remaining_candidates",
        "source_selection", "added_edges", "pruned_edges",
        "outgoing_scores", "incoming_scores",
    }
    assert expected.issubset(keys)


def test_topic_add_outgoing_skips_edges_that_create_cycle():
    import networkx as nx

    cc = make_cc_topic(N=3)

    cc.graph_state = nx.DiGraph()
    cc.graph_state.add_nodes_from(range(cc.N))
    cc.graph_state.add_edge(1, 0)

    cc.topological_order = [0]
    cc.candidates = [1, 2]

    table = {
        (1, frozenset()): 1000.0,
        (1, frozenset({0})): 0.0,
        (2, frozenset()): 1000.0,
        (2, frozenset({0})): 0.0,
    }
    patch_fake_scorer(cc, table)

    added_edges, all_scores = cc._graph_search_topological_add_outgoing(source=0)

    # 0->1 must be skipped because it would create a cycle with 1->0
    assert (0, 1) not in cc.graph_state.edges()
    assert all(e["from"] != 0 or e["to"] != 1 for e in added_edges)

    # 0->2 should be added (assuming it passes significance)
    assert (0, 2) in cc.graph_state.edges()


def test_topic_refine_extra_removes_redundant_parents():
    import networkx as nx

    cc = make_cc_topic(N=4)
    cc.graph_state = nx.DiGraph()
    cc.graph_state.add_nodes_from(range(cc.N))

    cc.graph_state.add_edge(0, 3)
    cc.graph_state.add_edge(1, 3)
    cc.graph_state.add_edge(2, 3)

    table = {
        (3, frozenset({0, 1, 2})): 100.0,
        (3, frozenset({0})): 100.0,
        (3, frozenset({1})): 999.0,
        (3, frozenset({2})): 999.0,
        (3, frozenset({0, 1})): 100.0,
        (3, frozenset({0, 2})): 100.0,
        (3, frozenset({1, 2})): 999.0,
    }
    patch_fake_scorer(cc, table)

    cc._graph_search_topological_refine_extra(min_parent_set_size=0)

    parents_after = set(cc.graph_state.predecessors(3))
    assert parents_after == {0}


def test_topic_meta_delta_and_incoming_pressure_are_consistent():
    import numpy as np
    import networkx as nx

    cc = make_cc_topic(N=3)
    cc.graph_state = nx.DiGraph()
    cc.graph_state.add_nodes_from(range(cc.N))
    candidates = [0, 1, 2]

    table = {}
    for j in candidates:
        table[(j, frozenset())] = 1000.0 + j

    table[(1, frozenset({0}))] = 900.0   # gain 100
    table[(2, frozenset({1}))] = 800.0   # gain 200
    table[(0, frozenset({2}))] = 950.0   # gain 50

    patch_fake_scorer(cc, table)

    source, meta = cc._graph_search_topological_next(candidates)

    improvement = np.array(meta["improvement_matrix"], dtype=float)
    delta = np.array(meta["delta_matrix"], dtype=float)
    incoming = np.array(meta["incoming_pressure"], dtype=float)

    for a in range(len(candidates)):
        for b in range(len(candidates)):
            if a == b:
                continue
            assert np.isclose(delta[a, b], improvement[a, b] - improvement[b, a])


    assert np.allclose(incoming, np.max(delta, axis=0))

    assert meta["source_idx"] == int(np.argmin(incoming))
    assert source == candidates[meta["source_idx"]]
