from __future__ import annotations

import pandas as pd
import pytest

from causalchange.domain.temporal import TemporalDomain, TimeGrid, util_changepoints_to_intervals


def test_temporal_domain_nodes_and_allowed_edges():
    X = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    domain = TemporalDomain(tau_max=2)

    X0 = domain.prepare_X(X)

    assert domain.variables(X0) == ["x", "y"]
    assert ("x", 0) in domain.nodes(X0)
    assert ("x", 2) in domain.nodes(X0)
    assert domain.allowed_edge(("x", 1), ("y", 0)) is True
    assert domain.allowed_edge(("x", 0), ("x", 0)) is False
    assert domain.allowed_edge(("x", 0), ("y", 1)) is False


def test_time_grid_helpers():
    X = pd.DataFrame({"x": [1, 2]})
    grid = TimeGrid(datasets={"a": X}, variables=["x"], context_col="context")

    assert grid.dataset_ids == ["a"]
    assert grid.n_contexts == 1
    assert grid.first_dataset() is X


def test_changepoints_to_intervals():
    assert util_changepoints_to_intervals(10, [3, 7]) == [(0, 3), (3, 7), (7, 10)]

    with pytest.raises(ValueError):
        util_changepoints_to_intervals(10, [0])
