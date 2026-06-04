from __future__ import annotations

import pytest

from causalchange.config.causal_change_config import (
    CausalChangeConfigTabular,
    CausalChangeConfigTemporal,
)
from causalchange.config.factory import ConfigFactory
from causalchange.core.types import (
    ChangepointMethod,
    ChangepointMode,
    ChangepointScope,
    DataMode,
    GraphSearch,
    MechanismClusteringMethod,
    MechanismClusteringScope,
    MixedSCMType,
    PostprocessingMode,
    ScoreType,
    StatisticalTestingMethod,
    TabularContextMethod,
    TabularContextMode,
)


def _make_config(**overrides):
    defaults = dict(
        cfg=None,
        data_mode=DataMode.TABULAR,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        mix_type=MixedSCMType.SKIP,
        context_mode=TabularContextMode.SKIP,
        context_combination_method=TabularContextMethod.SKIP,
        context_col=None,
        changepoint_mode=ChangepointMode.SKIP,
        changepoint_scope=ChangepointScope.SKIP,
        changepoint_method=ChangepointMethod.SKIP,
        clustering_scope=MechanismClusteringScope.SKIP,
        clustering_method=MechanismClusteringMethod.SKIP,
        testing_method=StatisticalTestingMethod.SKIP,
        postprocessing_mode=PostprocessingMode.SKIP,
        tau_max=None,
        d_min=30,
        max_iter=3,
        pelt_penalty="auto",
        mechanism_test_alpha=0.05,
        fixed_changepoints=None,
    )
    defaults.update(overrides)
    return ConfigFactory.make_causal_change_config(**defaults)


def test_factory_builds_tabular_config():
    cfg = _make_config()

    assert isinstance(cfg, CausalChangeConfigTabular)
    assert cfg.context_col == "context"


def test_factory_builds_temporal_config():
    cfg = _make_config(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        tau_max=2,
        postprocessing_mode=PostprocessingMode.EDGE_STRENGTHS,
    )

    assert isinstance(cfg, CausalChangeConfigTemporal)
    assert cfg.tau_max == 2
    assert cfg.postprocessing_mode is PostprocessingMode.EDGE_STRENGTHS


def test_factory_requires_tau_max_for_temporal_without_cfg():
    with pytest.raises(ValueError, match="tau_max"):
        _make_config(data_mode=DataMode.TIME, graph_search=GraphSearch.GLOBE)


def test_factory_returns_existing_config_unchanged():
    existing = CausalChangeConfigTabular(
        data_mode=DataMode.TABULAR,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
    )

    cfg = _make_config(cfg=existing, data_mode=DataMode.SKIP)
    assert cfg is existing
