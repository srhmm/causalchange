from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalchange.core.types import ClusteringMethod
from causalchange.discovery.context_combination import LINCContextCombination


def _toy_contexts():
    return {
        "a": pd.DataFrame({"ctx_id": [0, 0], "x": [0.0, 1.0]}),
        "b": pd.DataFrame({"ctx_id": [1, 1], "x": [2.0, 3.0]}),
        "c": pd.DataFrame({"ctx_id": [2, 2], "x": [4.0, 5.0]}),
    }


def _score_by_context_set(df: pd.DataFrame) -> float:
    ctx_set = frozenset(df["ctx_id"].unique())

    scores = {
        frozenset({0}): 10.0,
        frozenset({1}): 10.0,
        frozenset({2}): 10.0,
        frozenset({0, 1}): 15.0,  # positive gain: 10 + 10 - 15 = 5
        frozenset({0, 2}): 25.0,  # negative gain
        frozenset({1, 2}): 25.0,  # negative gain
        frozenset({0, 1, 2}): 40.0,
    }

    return scores[ctx_set]


def _as_group_sets(groups):
    return {frozenset(group) for group in groups}


def test_linc_components_merges_positive_gain_contexts():
    comb = LINCContextCombination(
        grouping=ClusteringMethod.COMPONENTS,
        gain_threshold=0.0,
        higher_is_better=False,
    )

    result = comb.combine_contexts(
        contexts=_toy_contexts(),
        effect="y",
        parents=("x",),
        score_ctx=_score_by_context_set,
    )

    assert result.total == 25.0
    assert result.diagnostics["method"] == "score-merge-components"
    assert _as_group_sets(result.diagnostics["groups"]) == {
        frozenset({"a", "b"}),
        frozenset({"c"}),
    }

    gain = result.diagnostics["gain_matrix"]
    assert gain.shape == (3, 3)
    assert np.allclose(gain, gain.T)
    assert comb.last_gain_contexts == ("a", "b", "c")


def test_linc_components_threshold_can_prevent_merge():
    comb = LINCContextCombination(
        grouping=ClusteringMethod.COMPONENTS,
        gain_threshold=5.1,
        higher_is_better=False,
    )

    result = comb.combine_contexts(
        contexts=_toy_contexts(),
        effect="y",
        parents=("x",),
        score_ctx=_score_by_context_set,
    )

    assert result.total == 30.0
    assert _as_group_sets(result.diagnostics["groups"]) == {
        frozenset({"a"}),
        frozenset({"b"}),
        frozenset({"c"}),
    }
    assert result.diagnostics["edges"] == []


def test_linc_components_does_not_merge_when_gain_equals_threshold():
    comb = LINCContextCombination(
        grouping=ClusteringMethod.COMPONENTS,
        gain_threshold=5.0,
        higher_is_better=False,
    )

    result = comb.combine_contexts(
        contexts=_toy_contexts(),
        effect="y",
        parents=("x",),
        score_ctx=_score_by_context_set,
    )

    assert result.total == 30.0
    assert _as_group_sets(result.diagnostics["groups"]) == {
        frozenset({"a"}),
        frozenset({"b"}),
        frozenset({"c"}),
    }
    assert result.diagnostics["edges"] == []


def test_linc_components_merges_when_gain_exceeds_threshold():
    comb = LINCContextCombination(
        grouping=ClusteringMethod.COMPONENTS,
        gain_threshold=4.9,
        higher_is_better=False,
    )

    result = comb.combine_contexts(
        contexts=_toy_contexts(),
        effect="y",
        parents=("x",),
        score_ctx=_score_by_context_set,
    )

    assert result.total == 25.0
    assert _as_group_sets(result.diagnostics["groups"]) == {
        frozenset({"a", "b"}),
        frozenset({"c"}),
    }
    assert result.diagnostics["edges"] == [("a", "b")]


def test_linc_agglomerative_merges_best_positive_gain_pair():
    comb = LINCContextCombination(
        grouping=ClusteringMethod.AGGLOMERATIVE,
        gain_threshold=0.0,
        higher_is_better=False,
    )

    result = comb.combine_contexts(
        contexts=_toy_contexts(),
        effect="y",
        parents=("x",),
        score_ctx=_score_by_context_set,
    )

    assert result.total == 25.0
    assert result.diagnostics["method"] == "score-merge-agglomerative"
    assert _as_group_sets(result.diagnostics["groups"]) == {
        frozenset({"a", "b"}),
        frozenset({"c"}),
    }


def test_linc_single_context_returns_single_group_score():
    comb = LINCContextCombination(
        grouping=ClusteringMethod.COMPONENTS,
        gain_threshold=0.0,
        higher_is_better=False,
    )

    contexts = {"a": _toy_contexts()["a"]}

    result = comb.combine_contexts(
        contexts=contexts,
        effect="y",
        parents=("x",),
        score_ctx=_score_by_context_set,
    )

    assert result.total == 10.0
    assert result.diagnostics["groups"] == [frozenset({"a"})]
    assert result.diagnostics["ctx_scores"] == {"a": 10.0}


def test_linc_empty_contexts_are_rejected():
    comb = LINCContextCombination(
        grouping=ClusteringMethod.COMPONENTS,
        gain_threshold=0.0,
        higher_is_better=False,
    )

    with pytest.raises(ValueError, match="at least one context"):
        comb.combine_contexts(
            contexts={},
            effect="y",
            parents=("x",),
            score_ctx=_score_by_context_set,
        )
