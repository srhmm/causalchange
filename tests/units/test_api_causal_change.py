import pandas as pd
import pytest

from causalchange.causal_change import CausalChange
from causalchange.config.cc_config import CausalChangeConfig
from causalchange.config.cc_types import (
    ContextAggregation,
    DataMode,
    GraphSearch,
    ScoreType,
)


def make_iid_df():
    return pd.DataFrame(
        {
            "x0": [0.0, 1.0, 2.0, 3.0, 4.0],
            "x1": [0.1, 1.1, 2.1, 3.1, 4.1],
            "x2": [1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )


def make_context_df():
    df = make_iid_df()
    df["context"] = [0, 0, 0, 1, 1]
    return df


@pytest.mark.parametrize(
    "data_mode, aggregation, X",
    [
        (DataMode.IID, ContextAggregation.SKIP, make_iid_df()),
        (DataMode.CONTEXTS, ContextAggregation.CHAIN, make_context_df()),
        (DataMode.CONTEXTS, ContextAggregation.LINC, make_context_df()),
    ],
)
def test_causalchange_public_api_smoke(data_mode, aggregation, X):
    cc = CausalChange(
        data_mode=data_mode,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        aggregation=aggregation,
        context_col="context" if data_mode.is_context() else None,
    )

    assert cc.data_mode == data_mode
    assert cc.graph_search == GraphSearch.TOPIC
    assert cc.score_type == ScoreType.LIN
    assert cc.aggregation == aggregation
    assert not cc.fitted_graph

    fitted = cc.fit(X)

    assert fitted is cc
    assert cc.fitted_graph
    assert cc.graph_ is not None
    assert cc.result_ is not None
    assert cc.graph is cc.graph_
    assert cc.get_result() is cc.result_
    assert isinstance(cc.topological_order_, list)
    assert isinstance(cc.history_, list)
    assert cc.graph_.is_directed()

    expected_nodes = [c for c in X.columns if c != "context"]
    assert set(cc.graph_.nodes) == set(expected_nodes)


def test_causalchange_rejects_cfg_and_individual_args():
    cfg = CausalChangeConfig(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.SKIP,
    )

    with pytest.raises(ValueError, match="Pass either cfg"):
        CausalChange(
            cfg=cfg,
            data_mode=DataMode.IID,
        )


def test_context_mode_requires_context_col():
    with pytest.raises(ValueError, match="context_col is required"):
        CausalChange(
            data_mode=DataMode.CONTEXTS,
            graph_search=GraphSearch.TOPIC,
            score_type=ScoreType.LIN,
            aggregation=ContextAggregation.CHAIN,
        )


def test_missing_context_column_errors_at_fit():
    cc = CausalChange(
        data_mode=DataMode.CONTEXTS,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        aggregation=ContextAggregation.CHAIN,
        context_col="context",
    )

    with pytest.raises(ValueError, match="requires context column"):
        cc.fit(make_iid_df())


def test_score_before_fit_errors():
    cc = CausalChange(
        data_mode=DataMode.IID,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
    )

    with pytest.raises(RuntimeError, match="Call fit"):
        cc.score("x0", ())
