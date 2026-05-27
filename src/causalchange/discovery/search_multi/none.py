from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AggregationResult:
    total: float
    diagnostics: dict[str, Any]


class NoAggregation:
    """ "Aggregation" for single-context"""

    def aggregate(
        self,
        *,
        contexts: dict[Hashable, pd.DataFrame],
        effect: Any,
        parents: tuple[Any, ...],
        score_ctx: Callable[[pd.DataFrame], float],
    ) -> AggregationResult:
        if len(contexts) != 1:
            raise ValueError(
                f"NoAggregation expects exactly one context, got {len(contexts)}. "
                "Use ContextAggregation.CHAIN or ContextAggregation.LINC for multi-context data."
            )

        ctx, df = next(iter(contexts.items()))
        score = float(score_ctx(df))

        return AggregationResult(
            total=score,
            diagnostics={
                "mode": "none",
                "context": ctx,
                "effect": effect,
                "parents": parents,
                "score": score,
            },
        )
