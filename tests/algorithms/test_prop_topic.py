# tests/test_prop_topic.py
import numpy as np
import networkx as nx

from hypothesis import given, settings, strategies as st

from causalchange.scoring.fit_cond_mixture import MixingType
from src.causalchange.cc_types import DataMode, GraphSearch, GPType
from src.causalchange.causal_change import CausalChange

from tests.utils.fake_edge_memoized_table import FakeEdgeMemoizedTable

"""Tests for TOPIC with fake scoring to test expected behavior"""

def make_cc_topic(N: int):
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

    # Deterministic significance: only strictly positive gains count
    cc.is_score_insignificant = lambda gain: gain <= 0.0
    return cc


def patch_fake_scorer(cc: CausalChange, table: dict):
    cc.edges_state = FakeEdgeMemoizedTable(
        cc.X, cc.data_mode, cc.score_type, cc.mixing_type,
        table=table,
        base=1000.0,
        penalty=10.0,
    )


@st.composite
def topic_score_tables(draw, N_min=3, N_max=6):
    """
    Provide enough scores for TOPIC source selection and outgoing adds:
      score([]->j) and score([i]->j) for all i!=j.

    For TOPIC _graph_search_topological_next at start, graph has no parents,
    so improvement(cause->effect) uses only those singleton vs empty scores.
    """
    N = draw(st.integers(min_value=N_min, max_value=N_max))

    base_scores = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
            min_size=N, max_size=N
        )
    )

    table = {}
    for j in range(N):
        table[(j, frozenset())] = float(base_scores[j])

    for j in range(N):
        for i in range(N):
            if i == j:
                continue
            table[(j, frozenset({i}))] = float(draw(
                st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False)
            ))

    return N, table


def expected_topic_source_from_table(N: int, table: dict, candidates: list[int]):
    """
    Replicates TOPIC's choice:
      improvement[cause,effect] = gain = score([]->effect) - score([cause]->effect)
      delta = improvement - improvement.T
      incoming_pressure[effect] = max_cause delta[cause,effect]
      source = argmin incoming_pressure
    """
    idx = {c: k for k, c in enumerate(candidates)}
    m = len(candidates)

    improvement = np.zeros((m, m), dtype=float)
    for cause in candidates:
        for eff in candidates:
            if cause == eff:
                continue
            s0 = table[(eff, frozenset())]
            s1 = table[(eff, frozenset({cause}))]
            improvement[idx[cause], idx[eff]] = s0 - s1

    delta = improvement - improvement.T
    np.fill_diagonal(delta, -np.inf)
    incoming_pressure = np.max(delta, axis=0)
    source_idx = int(np.argmin(incoming_pressure))
    return candidates[source_idx]


@settings(max_examples=200, deadline=None)
@given(topic_score_tables())
def test_topic_source_selection_matches_incoming_pressure_rule(data):
    N, table = data
    cc = make_cc_topic(N)
    patch_fake_scorer(cc, table)

    candidates = list(range(N))
    source, meta = cc._graph_search_topological_next(candidates)

    expected = expected_topic_source_from_table(N, table, candidates)
    assert source == expected


@settings(max_examples=200, deadline=None)
@given(topic_score_tables())
def test_topic_add_outgoing_adds_exactly_positive_gain_edges(data):
    """
    Property:
      With deterministic significance (gain>0),
      add_outgoing(source) adds exactly edges source->node where gain>0,
      assuming no cycle constraints block them (empty graph => no cycles).
    """
    N, table = data
    cc = make_cc_topic(N)
    patch_fake_scorer(cc, table)

    # Pick the TOPIC source for this table/candidates
    candidates = list(range(N))
    source, _ = cc._graph_search_topological_next(candidates)

    # Simulate TOPIC iteration state
    cc.candidates = [n for n in candidates if n != source]
    cc.topological_order = [source]

    added_edges, all_scores = cc._graph_search_topological_add_outgoing(source)

    expected_edges = set()
    for node in cc.candidates:
        gain = table[(node, frozenset())] - table[(node, frozenset({source}))]
        if gain > 0.0:
            expected_edges.add((source, node))

    got_edges = set((e["from"], e["to"]) for e in added_edges)
    assert got_edges == expected_edges

    # sanity: graph_state contains those edges
    assert set(cc.graph_state.edges()) == expected_edges


@settings(max_examples=100, deadline=None)
@given(topic_score_tables())
def test_topic_history_shape_invariants(data):
    """
    Property:
      Running full TOPIC ordering produces:
        - topic_history length == N
        - each entry contains required keys
        - topological_order grows by 1 each iteration
    """
    N, table = data
    cc = make_cc_topic(N)
    patch_fake_scorer(cc, table)

    # Run the full TOPIC graph search
    cc._graph_search_topological()

    assert len(cc.topic_history) == N

    required = {
        "iteration", "source", "topological_order", "remaining_candidates",
        "source_selection", "added_edges", "pruned_edges",
        "outgoing_scores", "incoming_scores"
    }

    for k, entry in enumerate(cc.topic_history, start=1):
        assert required.issubset(set(entry.keys()))
        assert entry["iteration"] == k
        assert len(entry["topological_order"]) == k
        assert len(entry["remaining_candidates"]) == N - k
