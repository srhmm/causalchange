from __future__ import annotations

import pytest
from pydantic import ValidationError

from causalchange.config.causal_change_config import (
    CausalChangeConfigTabular,
    CausalChangeConfigTemporal,
)
from causalchange.core.types import (
    ChangepointMode,
    DataMode,
    GPType,
    GraphSearch,
    MixedSCMType,
    ScoreType,
    TabularContextMode,
)


def test_valid_tabular_config():
    cfg = CausalChangeConfigTabular(
        data_mode=DataMode.TABULAR,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
    )

    assert cfg.context_col == "context"
    assert cfg.mix_type is MixedSCMType.SKIP


def test_valid_temporal_config():
    cfg = CausalChangeConfigTemporal(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=ScoreType.LIN,
        tau_max=2,
    )

    assert cfg.tau_max == 2
    assert cfg.changepoint_mode is ChangepointMode.SKIP


def test_tabular_config_rejects_temporal_data_mode():
    with pytest.raises(ValidationError):
        CausalChangeConfigTabular(
            data_mode=DataMode.TIME,
            graph_search=GraphSearch.GLOBE,
            score_type=ScoreType.LIN,
        )


def test_temporal_fixed_changepoints_require_oracle_mode():
    with pytest.raises(ValidationError):
        CausalChangeConfigTemporal(
            data_mode=DataMode.TIME,
            graph_search=GraphSearch.GLOBE,
            score_type=ScoreType.LIN,
            fixed_changepoints=[10],
        )


def test_context_detection_requires_mixing_type():
    with pytest.raises(ValidationError):
        CausalChangeConfigTabular(
            data_mode=DataMode.TABULAR,
            graph_search=GraphSearch.TOPIC,
            score_type=ScoreType.LIN,
            context_mode=TabularContextMode.DETECT,
        )


def test_gp_score_type_must_be_concrete():
    with pytest.raises(ValidationError):
        CausalChangeConfigTabular(
            data_mode=DataMode.TABULAR,
            graph_search=GraphSearch.TOPIC,
            score_type=ScoreType.GP,
        )

    cfg = CausalChangeConfigTemporal(
        data_mode=DataMode.TIME,
        graph_search=GraphSearch.GLOBE,
        score_type=GPType.FOURIER,
    )
    assert cfg.score_type is GPType.FOURIER
