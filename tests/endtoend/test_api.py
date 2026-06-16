from __future__ import annotations

import importlib
from typing import get_args

import pytest

import causalchange.api as api
from causalchange import CMM, Linc, SpaceTime, Topic
from causalchange.api import (
    ChangepointMethodName,
    ChangepointModeName,
    ChangepointScopeName,
    ClusteringMethodName,
    ClusteringScopeName,
    DataName,
    MixTypeName,
    PostprocessingName,
    ScoreName,
    TestingMethodName,
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
    PostprocessingMode,
    ScoreType,
    StatisticalTestingMethod,
    TabularContextMethod,
    TabularContextMode,
)


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


def test_public_api_exports_only_public_wrappers_and_literal_aliases():
    assert "Topic" in api.__all__
    assert "Linc" in api.__all__
    assert "CMM" in api.__all__
    assert "SpaceTime" in api.__all__

    assert "TopicConfig" not in api.__all__
    assert "LincConfig" not in api.__all__
    assert "CMMConfig" not in api.__all__
    assert "SpaceTimeConfig" not in api.__all__


def test_public_literals_match_supported_enum_values():
    assert set(get_args(ScoreName)) == {m.value for m in ScoreType if m != ScoreType.SKIP}

    assert set(get_args(PostprocessingName)) == {
        PostprocessingMode.SKIP.value,
        PostprocessingMode.EDGE_STRENGTHS.value,
    }

    assert set(get_args(MixTypeName)) == {
        MixedSCMType.LIN.value,
        MixedSCMType.QUADRATIC.value,
        MixedSCMType.CUBIC.value,
        MixedSCMType.N_SPLINE.value,
        MixedSCMType.B_SPLINE.value,
    }

    assert set(get_args(DataName)) == {
        DataMode.TIME.value,
        DataMode.TIME_CONTEXTS.value,
    }

    assert set(get_args(ChangepointModeName)) == {m.value for m in ChangepointMode}
    assert set(get_args(ChangepointScopeName)) == {m.value for m in ChangepointScope}
    assert set(get_args(ChangepointMethodName)) == {m.value for m in ChangepointMethod}
    assert set(get_args(ClusteringScopeName)) == {m.value for m in MechanismClusteringScope}
    assert set(get_args(ClusteringMethodName)) == {m.value for m in MechanismClusteringMethod}
    assert set(get_args(TestingMethodName)) == {m.value for m in StatisticalTestingMethod}


def test_topic_wrapper_builds_expected_config():
    est = Topic(score_type="lin", postprocessing_mode="skip", seed=123)

    assert est.public_config_.data_mode is DataMode.TABULAR
    assert est.public_config_.graph_search is GraphSearch.TOPIC
    assert est.public_config_.score_type is ScoreType.LIN
    assert est.public_config_.postprocessing_mode is PostprocessingMode.SKIP
    assert est.public_config_.seed == 123


def test_linc_wrapper_builds_expected_config():
    est = Linc(score_type="gam", context_col="env", seed=123)

    assert est.public_config_.data_mode is DataMode.TAB_CONTEXTS
    assert est.public_config_.graph_search is GraphSearch.TOPIC
    assert est.public_config_.score_type is ScoreType.GAM
    assert est.public_config_.context_col == "env"
    assert est.public_config_.context_mode is TabularContextMode.ORACLE
    assert est.public_config_.context_combination_method is TabularContextMethod.LINC
    assert est.public_config_.seed == 123


def test_cmm_wrapper_builds_expected_config():
    est = CMM(mix_type="quadratic", k_max=3, seed=123)

    assert est.public_config_.data_mode is DataMode.TABULAR
    assert est.public_config_.graph_search is GraphSearch.TOPIC
    assert est.public_config_.score_type is ScoreType.LIN
    assert est.public_config_.context_mode is TabularContextMode.SKIP
    assert est.public_config_.mix_type is MixedSCMType.QUADRATIC
    assert est.public_config_.score_kwargs["k_max"] == 3
    assert est.public_config_.seed == 123

    assert "lambda_mix" not in est.public_config_.score_kwargs
    assert "hybrid_mixing" not in est.public_config_.score_kwargs


def test_cmm_wrapper_rejects_removed_public_parameters():
    with pytest.raises(TypeError):
        CMM(score_type="gam")  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        CMM(lambda_mix=1.0)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        CMM(hybrid_mixing=True)  # type: ignore[call-arg]


def test_spacetime_wrapper_builds_expected_config():
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
        seed=123,
    )

    assert est.public_config_.data_mode is DataMode.TIME
    assert est.public_config_.graph_search is GraphSearch.GLOBE
    assert est.public_config_.score_type is ScoreType.LIN
    assert est.public_config_.tau_max == 1
    assert est.public_config_.changepoint_mode is ChangepointMode.SKIP
    assert est.public_config_.changepoint_scope is ChangepointScope.SKIP
    assert est.public_config_.changepoint_method is ChangepointMethod.SKIP
    assert est.public_config_.clustering_scope is MechanismClusteringScope.SKIP
    assert est.public_config_.clustering_method is MechanismClusteringMethod.SKIP
    assert est.public_config_.testing_method is StatisticalTestingMethod.SKIP
    assert est.public_config_.seed == 123
