from __future__ import annotations

import pytest
from pydantic import ValidationError

from causalchange.config.causal_change_config import (
    CausalChangeConfigTabular,
    CausalChangeConfigTemporal,
)
from causalchange.core.types import (
    ChangepointMethod,
    ChangepointMode,
    ChangepointScope,
    DataMode,
    GraphSearch,
    MechanismClusteringMethod,
    MechanismClusteringScope,
    MixedSCMType,
    ScoreType,
    StatisticalTestingMethod,
    TabularContextMode,
)


def test_tabular_config_rejects_temporal_data_modes():
    for data_mode in (DataMode.TIME, DataMode.TIME_CONTEXTS):
        with pytest.raises(ValidationError):
            CausalChangeConfigTabular(
                data_mode=data_mode,
                graph_search=GraphSearch.TOPIC,
                score_type=ScoreType.LIN,
            )


def test_temporal_config_rejects_tabular_data_modes():
    for data_mode in (DataMode.TABULAR, DataMode.TAB_CONTEXTS):
        with pytest.raises(ValidationError):
            CausalChangeConfigTemporal(
                data_mode=data_mode,
                graph_search=GraphSearch.GLOBE,
                score_type=ScoreType.LIN,
            )


def test_temporal_skip_changepoints_requires_skip_scope_and_method():
    with pytest.raises(ValidationError):
        CausalChangeConfigTemporal(
            data_mode=DataMode.TIME,
            graph_search=GraphSearch.GLOBE,
            score_type=ScoreType.LIN,
            changepoint_mode=ChangepointMode.SKIP,
            changepoint_scope=ChangepointScope.GLOBAL,
            changepoint_method=ChangepointMethod.PELT,
        )


def test_temporal_skip_clustering_requires_skip_method_and_testing():
    with pytest.raises(ValidationError):
        CausalChangeConfigTemporal(
            data_mode=DataMode.TIME,
            graph_search=GraphSearch.GLOBE,
            score_type=ScoreType.LIN,
            changepoint_mode=ChangepointMode.SKIP,
            changepoint_scope=ChangepointScope.SKIP,
            changepoint_method=ChangepointMethod.SKIP,
            clustering_scope=MechanismClusteringScope.SKIP,
            clustering_method=MechanismClusteringMethod.TESTING,
            testing_method=StatisticalTestingMethod.KERNEL,
        )


def test_temporal_positive_integer_parameters_are_validated():
    with pytest.raises(ValidationError):
        CausalChangeConfigTemporal(
            data_mode=DataMode.TIME,
            graph_search=GraphSearch.GLOBE,
            score_type=ScoreType.LIN,
            tau_max=0,
        )

    with pytest.raises(ValidationError):
        CausalChangeConfigTemporal(
            data_mode=DataMode.TIME,
            graph_search=GraphSearch.GLOBE,
            score_type=ScoreType.LIN,
            d_min=0,
        )

    with pytest.raises(ValidationError):
        CausalChangeConfigTemporal(
            data_mode=DataMode.TIME,
            graph_search=GraphSearch.GLOBE,
            score_type=ScoreType.LIN,
            max_iter=0,
        )


def test_tabular_context_detection_requires_supported_context_setup():
    with pytest.raises(ValidationError):
        CausalChangeConfigTabular(
            data_mode=DataMode.TABULAR,
            graph_search=GraphSearch.TOPIC,
            score_type=ScoreType.LIN,
            context_mode=TabularContextMode.DETECT,
        )


def test_tabular_config_accepts_plain_topic_setup():
    cfg = CausalChangeConfigTabular(
        data_mode=DataMode.TABULAR,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        context_mode=TabularContextMode.SKIP,
        mix_type=MixedSCMType.SKIP,
    )

    assert cfg.data_mode is DataMode.TABULAR
    assert cfg.graph_search is GraphSearch.TOPIC
    assert cfg.mix_type is MixedSCMType.SKIP


def test_tabular_config_accepts_cmm_setup():
    cfg = CausalChangeConfigTabular(
        data_mode=DataMode.TABULAR,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
        context_mode=TabularContextMode.SKIP,
        mix_type=MixedSCMType.LIN,
        score_kwargs={"k_max": 2},
    )

    assert cfg.mix_type is MixedSCMType.LIN
    assert cfg.score_kwargs["k_max"] == 2
