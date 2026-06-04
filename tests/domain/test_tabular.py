from __future__ import annotations

import pandas as pd

from causalchange.domain.tabular import TabularDomain


def test_tabular_domain_nodes_and_edges():
    X = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    domain = TabularDomain()

    assert domain.prepare_X(X) is X
    assert domain.nodes(X) == ["x", "y"]
    assert domain.candidates(X) == ["x", "y"]
    assert domain.allowed_edge("x", "y") is True
    assert domain.allowed_edge("x", "x") is False
