from __future__ import annotations

import importlib

import pytest

from causalchange import Topic
from causalchange.causal_change import CausalChange
from causalchange.config.causal_change_config import CausalChangeConfigTabular
from causalchange.core.types import DataMode, GraphSearch, ScoreType


def test_core_modules_import():
    modules = [
        "causalchange.causal_change",
        "causalchange.config.causal_change_config",
        "causalchange.core.results",
        "causalchange.core.types",
        "causalchange.domain.tabular",
        "causalchange.domain.temporal",
        "causalchange.engines.factory",
        "causalchange.engines.tabular",
        "causalchange.engines.temporal",
        "causalchange.posthoc.edge_strengths",
    ]
    for module in modules:
        importlib.import_module(module)


def test_topic_instantiates_tabular():
    cfg = CausalChangeConfigTabular(
        data_mode=DataMode.TABULAR,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
    )

    model = CausalChange(cfg)

    assert model.cfg.data_mode is DataMode.TABULAR
    assert model.cfg.graph_search is GraphSearch.TOPIC


def test_causal_change_instantiates_tabular():
    model = Topic(score_type="lin")

    assert model.cfg.data_mode is DataMode.TABULAR
    assert model.cfg.graph_search is GraphSearch.TOPIC


def test_accessing_result_before_fit_raises():
    model = Topic(score_type="lin")

    with pytest.raises(RuntimeError, match="Call fit"):
        _ = model.graph_
