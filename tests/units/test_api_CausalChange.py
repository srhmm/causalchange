import logging

import pytest

from benchmarks.run_methods import run_sampling
from causalchange.config.cc_types import (
    DataMode,
    GraphSearch,
    ScoreType,
    ContextAggregation,
)
from causalchange.causal_change import CausalChange
from endtoend.test_endtoend import _get_config_for_data_and_algo


@pytest.mark.parametrize(
    "data_mode",
    [
        DataMode.IID,
        DataMode.CONTEXTS,
        DataMode.MIXED,
        DataMode.TIME,
        DataMode.TIME_CONTEXTS,
    ],
)
@pytest.mark.parametrize("graph_search", [GraphSearch.TOPIC])
@pytest.mark.parametrize(
    "context_aggregation",
    [ContextAggregation.SKIP, ContextAggregation.CHAIN, ContextAggregation.LINC],
)
def test_api_causalchange(
    data_mode: DataMode,
    graph_search: GraphSearch,
    context_aggregation: ContextAggregation,
):
    """test usage with each valid combo of graph search and data mode"""

    if not graph_search.is_compatible_with(
        data_mode
    ) or not context_aggregation.is_compatible_with(data_mode):
        pytest.skip(
            f"{graph_search}&{context_aggregation} not compatible with data_mode {data_mode}"
        )
    if data_mode in [DataMode.MIXED, DataMode.TIME, DataMode.TIME_CONTEXTS]:
        pytest.skip(f"{data_mode}")
    score_type = ScoreType.LIN
    lg = logging.basicConfig(level=logging.DEBUG)
    vb = 1

    cc = CausalChange(
        data_mode=data_mode,
        graph_search=graph_search,
        score_type=score_type,
        context_aggregation=context_aggregation,
        lg=lg,
        vb=vb,
    )
    assert cc.data_mode == data_mode
    assert cc.graph_search == graph_search
    assert cc.score_type == score_type
    cc._info("test")
    assert not cc.fitted_graph

    cfg = _get_config_for_data_and_algo(data_mode, graph_search, score_type)
    df, true_g = run_sampling(cfg.data)

    cc.fit(df)

    if data_mode.is_temporal():
        assert len(cc.graph_.nodes) == cfg.data.n_nodes * (cfg.data.tau_max + 1)
    else:
        assert len(cc.graph_.nodes) == cfg.data.n_nodes
    assert cc.graph_.is_directed()
    assert cc.fitted_graph


def test_api_causalchange_default():
    """test usage with default parameters"""

    cc = CausalChange(
        data_mode=DataMode.IID, graph_search=GraphSearch.TOPIC, score_type=ScoreType.GAM
    )
    default_score_type = ScoreType.GAM
    default_data_mode = DataMode.IID
    default_graph_search = GraphSearch.TOPIC
    assert cc.data_mode == default_data_mode
    assert cc.graph_search == default_graph_search
    assert cc.score_type == default_score_type

    cc._info("test")
    assert not cc.fitted_graph

    cfg = _get_config_for_data_and_algo(
        default_data_mode, default_graph_search, default_score_type
    )
    df, true_g = run_sampling(cfg.data)

    cc.fit(df)

    assert len(cc.graph_.nodes) == cfg.data.n_nodes
    assert cc.graph_.is_directed()
    assert cc.fitted_graph
