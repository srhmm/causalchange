import pytest

from causalchange.config.cc_config import CausalChangeConfig, ChangepointMode, SpaceTimeConfig
from causalchange.config.cc_types import ContextAggregation, DataMode, GraphSearch, ScoreType


def test_temporal_config_requires_spacetime():
    with pytest.raises(ValueError, match="spacetime"):
        CausalChangeConfig(
            data_mode=DataMode.TIME,
            graph_search=GraphSearch.GLOBE,
            score_type=ScoreType.LIN,
            aggregation=ContextAggregation.SKIP,
        )


def test_non_temporal_config_rejects_spacetime():
    with pytest.raises(ValueError, match="spacetime"):
        CausalChangeConfig(
            data_mode=DataMode.IID,
            graph_search=GraphSearch.TOPIC,
            score_type=ScoreType.LIN,
            aggregation=ContextAggregation.SKIP,
            spacetime=SpaceTimeConfig(tau_max=2),
        )


def test_time_contexts_rejects_linc_aggregation():
    with pytest.raises(ValueError, match="aggregation|compatible"):
        CausalChangeConfig(
            data_mode=DataMode.TIME_CONTEXTS,
            graph_search=GraphSearch.TOPIC,
            score_type=ScoreType.LIN,
            aggregation=ContextAggregation.LINC,
            context_col="context",
            spacetime=SpaceTimeConfig(tau_max=2),
        )


def test_fixed_changepoints_requires_values():
    with pytest.raises(ValueError, match="fixed_changepoints"):
        SpaceTimeConfig(
            tau_max=2,
            changepoints=ChangepointMode.FIXED,
        )


def test_fixed_changepoints_only_valid_for_fixed_mode():
    with pytest.raises(ValueError, match="fixed_changepoints"):
        SpaceTimeConfig(
            tau_max=2,
            fixed_changepoints=[10, 20],
        )


def test_spacetime_config_accepts_named_pelt_penalties():
    for penalty in ["bic", "mbic", "auto"]:
        cfg = SpaceTimeConfig(
            tau_max=2,
            changepoints=ChangepointMode.DETECT,
            pelt_penalty=penalty,
        )

        assert cfg.pelt_penalty == penalty


def test_spacetime_config_rejects_invalid_pelt_penalty():
    with pytest.raises(ValueError, match="pelt_penalty"):
        SpaceTimeConfig(
            tau_max=2,
            changepoints=ChangepointMode.DETECT,
            pelt_penalty="bad",
        )
