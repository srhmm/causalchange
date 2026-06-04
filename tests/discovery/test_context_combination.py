from __future__ import annotations

import pandas as pd
import pytest

from causalchange.discovery.context_combination import SkipCombination


def test_skip_combination_scores_single_context():
    comb = SkipCombination()
    contexts = {0: pd.DataFrame({"x": [1, 2]})}

    result = comb.combine_contexts(
        contexts=contexts,
        effect="x",
        parents=(),
        score_ctx=lambda df: float(len(df)),
    )

    assert result.total == 2.0
    assert result.diagnostics["mode"] == "none"


def test_skip_combination_rejects_multiple_contexts():
    comb = SkipCombination()
    contexts = {0: pd.DataFrame({"x": [1]}), 1: pd.DataFrame({"x": [2]})}

    with pytest.raises(ValueError, match="exactly one context"):
        comb.combine_contexts(
            contexts=contexts,
            effect="x",
            parents=(),
            score_ctx=lambda df: 0.0,
        )
