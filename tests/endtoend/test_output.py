from __future__ import annotations

import pandas as pd
import pytest

from causalchange import SpaceTime, Topic


def test_public_properties_raise_before_fit():
    est = Topic(score_type="lin")

    with pytest.raises(RuntimeError, match="fit"):
        _ = est.graph_

    with pytest.raises(RuntimeError, match="fit"):
        _ = est.history_

    with pytest.raises(RuntimeError, match="fit"):
        _ = est.edge_strengths_


def test_topic_public_properties_after_fit():
    X = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 2.0, 4.0, 6.0, 8.0],
        }
    )

    est = Topic(score_type="lin", seed=0).fit(X)

    assert est.graph_ is not None
    assert set(est.graph_.nodes()) == {"x", "y"}
    assert est.topological_order_ is not None
    assert est.history_ is not None
    assert est.get_result() is est.result_
    assert est.cmm_components_ is None


def test_spacetime_public_properties_after_skip_fit():
    X = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )

    est = SpaceTime(
        score_type="lin",
        data_mode="time",
        tau_max=1,
        changepoint_mode="skip",
        changepoint_scope="skip",
        changepoint_method="skip",
        clustering_scope="skip",
        clustering_method="skip",
        testing_method="skip",
        d_min=2,
        max_iter=1,
        seed=0,
    ).fit(X)

    assert est.graph_ is not None
    assert est.changepoints_ == []
    # assert est.changepoint_diagnostics_ is not None
    assert est.partitions_ is not None
    assert est.history_ is not None
