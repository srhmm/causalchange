import logging

import pytest

from benchmarks.run_methods import run_sampling
from causalchange.config._cc_types import DataMode, GraphSearch, ScoreType, MixingType
from causalchange.causal_change import CausalChange
from endtoend.test_endtoend import _get_config_for_data_and_algo


@pytest.mark.parametrize(
    "data_mode",
    [DataMode.IID, DataMode.CONTEXTS, DataMode.MIXED, DataMode.TIME, DataMode.TIME_CONTEXTS],
)
@pytest.mark.parametrize(
    "graph_search",
    [GraphSearch.TOPIC, GraphSearch.CHAIN]
)


def test_api_causalchange(data_mode: DataMode, graph_search: GraphSearch):
    """test usage with each valid combo of graph search and data mode"""

    if not graph_search.is_compatible_with(data_mode):
        pytest.skip(f"{graph_search} not compatible with data_mode {data_mode}")

    score_type = ScoreType.LIN
    lg = logging.basicConfig(level=logging.DEBUG)
    vb = 1

    if data_mode == DataMode.MIXED:
        mixing_type = MixingType.MIX_LIN

        cc = CausalChange(
            data_mode=data_mode,
            graph_search=graph_search,
            score_type=score_type,
            mixing_type=mixing_type,
            lg=lg,
            vb=vb,
        )
    else:
        cc = CausalChange(
            data_mode=data_mode,
            graph_search=graph_search,
            score_type=score_type,
            lg=lg,
            vb=vb,
        )
    assert cc.data_mode == data_mode
    assert cc.graph_search == graph_search
    assert cc.score_type == score_type
    cc._info("test")
    assert len(cc.graph_.nodes) == 0
    assert len(cc.graph_.edges) == 0
    assert cc.graph_.is_directed()
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

    cc = CausalChange( )
    default_score_type = ScoreType.GAM
    default_data_mode = DataMode.IID
    default_graph_search = GraphSearch.TOPIC
    assert cc.data_mode == default_data_mode
    assert cc.graph_search == default_graph_search
    assert cc.score_type == default_score_type

    cc._info("test")
    assert len(cc.graph_.nodes) == 0
    assert len(cc.graph_.edges) == 0
    assert cc.graph_.is_directed()
    assert not cc.fitted_graph


    cfg = _get_config_for_data_and_algo(default_data_mode, default_graph_search, default_score_type)
    df, true_g = run_sampling(cfg.data)

    cc.fit(df)

    assert len(cc.graph_.nodes) == cfg.data.n_nodes
    assert cc.graph_.is_directed()
    assert cc.fitted_graph