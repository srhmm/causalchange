import numpy as np
import pandas as pd
from endtoend.test_endtoend import _get_config_for_data_and_algo

from benchmarks.run_methods import run_sampling
from causalchange.config.cc_types import DataMode, GraphSearch, ScoreType
from causalchange.discovery.old._mixins import (
    LINCGroupingParams,
    LINCMixin,
    TabularScoreMixin,
)


class Host(LINCMixin):
    def __init__(self, *, grouping: LINCGroupingParams, score_higher_better: bool = True):
        super().__init__(grouping=grouping)
        self.data_mode = DataMode.CONTEXTS
        self.score_higher_better = score_higher_better
        self._current_df = None
        self._X_context = {}

    def _init_score(self, X: pd.DataFrame) -> None:
        self._current_df = X


def patch_parent_score(monkeypatch, fn):
    monkeypatch.setattr(TabularScoreMixin, "_score", fn, raising=True)


def test_components_no_contexts_returns_zero(monkeypatch):
    h = Host(grouping=LINCGroupingParams(method="components", gain_threshold=0.0))

    h._X_context = {}
    patch_parent_score(monkeypatch, lambda self, effect, parents: 123.0)

    assert h._score("y", []) == 0.0


def test_components_one_context_returns_context_score(monkeypatch):
    h = Host(grouping=LINCGroupingParams(method="components", gain_threshold=0.0))
    h._X_context = {"A": pd.DataFrame({"x": [1, 2]})}
    # Score = number of rows in current df
    patch_parent_score(monkeypatch, lambda self, effect, parents: float(len(self._current_df)))

    assert h._score("y", []) == 2.0


def test_components_edge_added_when_gain_exceeds_threshold_and_caches_gain(monkeypatch):
    h = Host(grouping=LINCGroupingParams(method="components", gain_threshold=0.5))
    h._X_context = {
        "A": pd.DataFrame({"x": [1, 2]}),
        "B": pd.DataFrame({"x": [3, 4]}),
    }

    def score(self, effect, parents):
        n = len(self._current_df)
        # Individual contexts: return 2
        # Pooled (n=4): return 10 to create gain 10 - (2+2) = 6 (> 0.5)
        return 10.0 if n == 4 else float(n)

    patch_parent_score(monkeypatch, score)

    total = h._score("y", [])

    # With an edge, union-find yields one component => one pooled score
    assert total == 10.0

    # Gain matrix / contexts cached
    assert h._last_gain_contexts == ("A", "B")
    assert h._last_gain_matrix.shape == (2, 2)
    assert np.allclose(h._last_gain_matrix, h._last_gain_matrix.T)
    assert h._last_gain_matrix[0, 1] > 0.5


def test_agglomerative_merges_until_threshold(monkeypatch):
    h = Host(grouping=LINCGroupingParams(method="agglomerative", gain_threshold=1.0))
    h._X_context = {
        "A": pd.DataFrame({"x": [1, 2]}),  # n=2
        "B": pd.DataFrame({"x": [3, 4]}),  # n=2
        "C": pd.DataFrame({"x": [5, 6]}),  # n=2
    }

    def score(self, effect, parents):
        n = len(self._current_df)
        # singletons n=2 => 2
        # any merged pair n=4 => 6 (gain = 6-(2+2)=2, significant)
        # all three n=6 => 7 (gain = 7-(6+2)=-1, not significant)
        if n == 4:
            return 6.0
        if n == 6:
            return 7.0
        return float(n)

    patch_parent_score(monkeypatch, score)

    total = h._score("y", [])

    # one merge gives score 6, plus remaining singleton 2 => 8
    assert total == 8.0


def test_transition_gain_respects_score_higher_better_false(monkeypatch):
    h = Host(
        grouping=LINCGroupingParams(method="components", gain_threshold=0.0),
        score_higher_better=False,
    )

    h._X_context = {
        "A": pd.DataFrame({"x": [1, 2]}),
        "B": pd.DataFrame({"x": [3, 4]}),
    }

    def score(self, effect, parents):
        n = len(self._current_df)
        # individual scores: 10 each (old total 20)
        # pooled score: 15 (better because lower is better)
        return 15.0 if n == 4 else 10.0

    patch_parent_score(monkeypatch, score)

    total = h._score("y", [])

    # With lower-better, gain = old - new = 20 - 15 = +5 => edge => pooled score
    assert total == 15.0


def test_extension_LINCMixin():
    linc_model = Host(grouping=LINCGroupingParams(method="components", gain_threshold=0.0))

    linc_model.score_higher_better = True
    old_score = 10
    new_score = 0
    assert linc_model._transition_gain(old_score, new_score) == new_score - old_score

    linc_model.score_higher_better = False
    old_score = 10
    new_score = 0
    assert linc_model._transition_gain(old_score, new_score) == old_score - new_score

    linc_model.score_higher_better = True

    old_score, new_score, new_better_score = (
        0,
        LINCGroupingParams.gain_threshold - 0.01,
        LINCGroupingParams.gain_threshold + 0.01,
    )
    assert not linc_model._score_significant(linc_model._transition_gain(old_score, new_score))
    assert linc_model._score_significant(linc_model._transition_gain(old_score, new_better_score))

    cfg = _get_config_for_data_and_algo(DataMode.CONTEXTS, GraphSearch.TOPIC, ScoreType.GAM)
    df, true_g = run_sampling(cfg.data)

    host = linc_model._host()
    X0 = host._init_contexts(df)
    host._init_score(X0)

    # smoke test for scoring
    for x in df.columns:
        for y in df.columns:
            if x == y:
                continue
            _ = linc_model._score(x, [y])

    # todo test scoring more?
