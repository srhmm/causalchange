from __future__ import annotations

from causalchange.core.types import (
    DataMode,
    MechanismClusteringScope,
    ScoreType,
)


def test_data_mode_helpers():
    assert DataMode.TABULAR.is_context() is False
    assert DataMode.TAB_CONTEXTS.is_context() is True
    assert DataMode.TIME.is_temporal() is True
    assert DataMode.TIME_CONTEXTS.is_context() is True
    assert DataMode.TIME_CONTEXTS.is_temporal() is True


def test_mechanism_clustering_scope_helpers():
    assert MechanismClusteringScope.SKIP.detects_contexts() is False
    assert MechanismClusteringScope.CONTEXTS.detects_contexts() is True
    assert MechanismClusteringScope.REGIMES.detects_regimes() is True
    assert MechanismClusteringScope.REGIMES_CONTEXTS.detects_contexts() is True
    assert MechanismClusteringScope.REGIMES_CONTEXTS.detects_regimes() is True


def test_lower_is_better_score_convention():
    assert ScoreType.LIN.higher_is_better is False
    assert ScoreType.LIN.transition_gain(old_score=10.0, new_score=7.0) == 3.0
    assert ScoreType.LIN.gain_is_better(3.0, 2.0) is True
    assert ScoreType.LIN.raw_score_is_better(2.0, 3.0) is True
