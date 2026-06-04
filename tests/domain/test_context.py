from __future__ import annotations

import pandas as pd
import pytest

from causalchange.domain.context import MultiContextDomain, SingleContextDomain


def test_single_context_domain_wraps_data():
    X = pd.DataFrame({"x": [1, 2]})
    domain = SingleContextDomain()

    assert domain.make_contexts(X) == {0: X}
    assert domain.prepare_X(X) is X


def test_multi_context_domain_splits_and_drops_context_column():
    X = pd.DataFrame({"context": ["a", "a", "b"], "x": [1, 2, 3]})
    domain = MultiContextDomain(context_col="context")

    contexts = domain.make_contexts(X)

    assert set(contexts) == {"a", "b"}
    assert "context" not in contexts["a"].columns
    assert domain.prepare_X(X).columns.tolist() == ["x"]


def test_multi_context_domain_requires_context_column():
    domain = MultiContextDomain(context_col="context")
    with pytest.raises(ValueError, match="context_col"):
        domain.prepare_X(pd.DataFrame({"x": [1, 2]}))
