from __future__ import annotations

import pandas as pd
import pytest

from causalchange import Linc, SpaceTime, Topic


def test_topic_check_input_accepts_array_like_input():
    est = Topic(score_type="lin")

    checked = est.check_input([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])

    assert isinstance(checked, pd.DataFrame)
    assert est.N == 3
    assert est.D == 2
    assert len(est.feature_cols_) == 2


def test_check_input_rejects_empty_dataframe():
    est = Topic(score_type="lin")

    with pytest.raises(ValueError, match="at least one row"):
        est.check_input(pd.DataFrame())


def test_topic_check_input_rejects_wrong_var_names_length():
    est = Topic(score_type="lin", var_nms=["x"])

    with pytest.raises(ValueError, match="node_nms"):
        est.check_input(pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 3.0]}))


def test_linc_check_input_requires_context_column():
    est = Linc(score_type="lin", context_col="context")

    with pytest.raises(ValueError, match="context_col"):
        est.check_input(pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 3.0]}))


def test_linc_check_input_rejects_nan_context_column():
    est = Linc(score_type="lin", context_col="context")

    with pytest.raises(ValueError, match="missing values"):
        est.check_input(
            pd.DataFrame(
                {
                    "x": [1.0, 2.0, 3.0],
                    "context": [0, None, 1],
                }
            )
        )


def test_linc_check_input_excludes_context_column_from_feature_count():
    est = Linc(score_type="lin", context_col="context")

    est.check_input(
        pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [2.0, 4.0, 6.0],
                "context": [0, 0, 1],
            }
        )
    )

    assert est.D == 2
    assert est.feature_cols_ == ["x", "y"]


def test_linc_check_input_rejects_only_context_column():
    est = Linc(score_type="lin", context_col="context")

    with pytest.raises(ValueError, match="No feature columns"):
        est.check_input(pd.DataFrame({"context": [0, 1, 1]}))


def test_spacetime_time_contexts_requires_context_column():
    est = SpaceTime(
        data_mode="time-contexts",
        score_type="lin",
        changepoint_mode="skip",
        changepoint_scope="skip",
        changepoint_method="skip",
        clustering_scope="skip",
        clustering_method="skip",
        testing_method="skip",
        context_col="context",
    )

    with pytest.raises(ValueError, match="context_col"):
        est.check_input(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
