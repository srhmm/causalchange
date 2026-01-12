from __future__ import annotations

import pandas as pd

from causalchange.discovery.domain.multi import MultiContextDomain

from causalchange.discovery.domain.single import SingleContextDomain


from causalchange.discovery.domain.tabular import TabularDomain
from causalchange.discovery.domain.temporal import TemporalDomain


def test_tabular_domain_nodes_candidates_and_allowed_edge():
    X = pd.DataFrame({"X0": [0.0, 1.0], "X1": [1.0, 2.0], "X2": [2.0, 3.0]})
    d = TabularDomain()
    X0 = d.prepare_X(X)

    assert list(X0.columns) == ["X0", "X1", "X2"]
    assert d.nodes(X0) == ["X0", "X1", "X2"]
    assert d.candidates(X0) == ["X0", "X1", "X2"]

    assert d.allowed_edge("X0", "X1") is True
    assert d.allowed_edge("X1", "X1") is False


def test_temporal_domain_nodes_candidates_and_allowed_edge_respects_lags_and_instantaneous_flag():
    X = pd.DataFrame({"X0": [0.0, 1.0, 2.0], "X1": [1.0, 2.0, 3.0]})

    d_no_inst = TemporalDomain(tau_max=2, allow_instantaneous=False)
    X0 = d_no_inst.prepare_X(X)

    nodes = d_no_inst.nodes(X0)
    assert ("X0", 0) in nodes and ("X1", 0) in nodes
    assert ("X0", 2) in nodes and ("X1", 2) in nodes

    # candidates are only lag-0 nodes (effects)
    candidates = d_no_inst.candidates(X0)
    assert set(candidates) == {("X0", 0), ("X1", 0)}

    assert d_no_inst.allowed_edge(("X0", 1), ("X1", 0)) is True
    assert d_no_inst.allowed_edge(("X0", 0), ("X1", 0)) is False
    assert d_no_inst.allowed_edge(("X0", 0), ("X1", 1)) is False

    d_inst = TemporalDomain(tau_max=1, allow_instantaneous=True)
    X1 = d_inst.prepare_X(X)
    assert d_inst.allowed_edge(("X0", 0), ("X1", 0)) is True


def test_single_context_provider_returns_single_context():
    X = pd.DataFrame({"X0": [0, 1, 2], "X1": [3, 4, 5]})
    p = SingleContextDomain()
    contexts = p.make_contexts(X)
    assert set(contexts.keys()) == {0}
    assert contexts[0].equals(X)


def test_multi_context_domain_splits_and_drops_context_column():
    X = pd.DataFrame(
        {
            "X0": [0, 1, 2, 3],
            "X1": [10, 11, 12, 13],
            "context": [0, 0, 1, 1],
        }
    )
    p = MultiContextDomain(context_col="context")
    contexts = p.make_contexts(X)

    assert set(contexts.keys()) == {0, 1}
    assert list(contexts[0].columns) == ["X0", "X1"]
    assert list(contexts[1].columns) == ["X0", "X1"]
    assert len(contexts[0]) == 2 and len(contexts[1]) == 2
