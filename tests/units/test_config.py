import pytest
from pydantic import ValidationError

from causalchange.config.cc_config import CausalChangeConfigTabular, ChangepointMode, CausalChangeConfigTime
from causalchange.config.cc_types import ContextMode, DataMode, GraphSearch, ScoreType


def test_temporal_config_requires_spacetime():
    with pytest.raises(ValueError, match="spacetime"):
        CausalChangeConfigTabular(
            data_mode=DataMode.TIME,
            graph_search=GraphSearch.GLOBE,
            score_type=ScoreType.LIN,
            aggregation=ContextMode.SKIP,
        )


def test_non_temporal_config_rejects_spacetime():
    with pytest.raises(ValueError, match="spacetime"):
        CausalChangeConfigTabular(
            data_mode=DataMode.IID,
            graph_search=GraphSearch.TOPIC,
            score_type=ScoreType.LIN,
            aggregation=ContextMode.SKIP,
            spacetime=CausalChangeConfigTime(tau_max=2),
        )


def test_time_contexts_rejects_linc_aggregation():
    with pytest.raises(ValueError, match="aggregation|compatible"):
        CausalChangeConfigTabular(
            data_mode=DataMode.TIME_CONTEXTS,
            graph_search=GraphSearch.TOPIC,
            score_type=ScoreType.LIN,
            aggregation=ContextMode.LINC,
            context_col="context",
            spacetime=CausalChangeConfigTime(tau_max=2),
        )


def test_fixed_changepoints_requires_values():
    with pytest.raises(ValueError, match="fixed_changepoints"):
        CausalChangeConfigTime(
            tau_max=2,
            changepoints=ChangepointMode.FIXED,
        )


def test_fixed_changepoints_only_valid_for_fixed_mode():
    with pytest.raises(ValueError, match="fixed_changepoints"):
        CausalChangeConfigTime(
            tau_max=2,
            fixed_changepoints=[10, 20],
        )


def test_spacetime_config_accepts_named_pelt_penalties():
    for penalty in ["bic", "mbic", "auto"]:
        cfg = CausalChangeConfigTime(
            tau_max=2,
            changepoints=ChangepointMode.DETECT,
            pelt_penalty=penalty,
        )

        assert cfg.pelt_penalty == penalty


def test_spacetime_config_rejects_invalid_pelt_penalty():
    with pytest.raises(ValueError, match="pelt_penalty"):
        CausalChangeConfigTime(
            tau_max=2,
            changepoints=ChangepointMode.DETECT,
            pelt_penalty="bad",
        )


def test_topic_is_compatible_with_tabular_modes():
    assert GraphSearch.TOPIC.is_compatible_with(DataMode.IID)
    assert GraphSearch.TOPIC.is_compatible_with(DataMode.CONTEXTS)


def test_globe_is_compatible_with_temporal_modes():
    assert GraphSearch.GLOBE.is_compatible_with(DataMode.TIME)
    assert GraphSearch.GLOBE.is_compatible_with(DataMode.TIME_CONTEXTS)


def test_topic_is_not_public_for_temporal_modes():
    with pytest.raises(ValidationError):
        CausalChangeConfigTabular(
            data_mode=DataMode.TIME,
            graph_search=GraphSearch.TOPIC,
            score_type=ScoreType.LIN,
            aggregation=ContextMode.SKIP,
            spacetime=CausalChangeConfigTime(
                tau_max=1,
                changepoints=ChangepointMode.NONE,
            ),
        )
