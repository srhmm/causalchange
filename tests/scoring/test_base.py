from __future__ import annotations

import pytest

from causalchange.config.causal_change_config import CausalChangeConfigTabular
from causalchange.core.types import DataMode, GraphSearch, ScoreType
from causalchange.scoring.base import BaseLocalScorer, SCMScore


class DummyLocalScorer(BaseLocalScorer):
    def local_score(self, *args, **kwargs) -> float:
        return 0.0


def _cfg():
    return CausalChangeConfigTabular(
        data_mode=DataMode.TABULAR,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.LIN,
    )


def test_base_local_scorer_delegates_score_convention():
    scorer = DummyLocalScorer(_cfg())

    assert scorer.higher_is_better is False
    assert scorer.transition_gain(10.0, 7.0) == 3.0
    assert scorer.gain_is_better(3.0, 2.0) is True
    assert scorer.raw_score_is_better(2.0, 3.0) is True

    with pytest.raises(RuntimeError, match="fit"):
        scorer.score_significant(1.0)

    scorer._set_global_n_samples(10)
    assert scorer.score_significant(10.0) is True


def test_low_level_scm_score_requires_fit_before_local_score():
    backend = SCMScore(data_mode=DataMode.TABULAR, score_type=ScoreType.LIN)
    with pytest.raises(RuntimeError, match="fit"):
        backend.local_score(j=0, pa=[])
