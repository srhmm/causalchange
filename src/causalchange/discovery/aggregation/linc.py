from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, Literal, Optional

import numpy as np
import pandas as pd

from causalchange.discovery.aggregation.utils.union_find import union_find_components


@dataclass(frozen=True)
class AggregationResult:
    total: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class LINCGroupingParams:
    method: Literal["components", "agglomerative"] = "components"
    gain_threshold: float = 0.0


class LINCAggregator:
    """
    Port of LINCMixin, but as a pure aggregator:
      - takes contexts dict
      - takes score_ctx(df) callback
      - returns total + diagnostics (gain matrix, groups, etc.)
    """

    def __init__(self, *, grouping: LINCGroupingParams, higher_is_better: bool):
        self.grouping = grouping
        self.higher_is_better = bool(higher_is_better)

        # last-run diagnostics (handy)
        self.last_gain_matrix: Optional[np.ndarray] = None
        self.last_gain_contexts: Optional[tuple[Hashable, ...]] = None

    def transition_gain(self, old_score: float, new_score: float) -> float:
        return (new_score - old_score) if self.higher_is_better else (old_score - new_score)

    def score_significant(self, gain: float) -> bool:
        return gain > float(self.grouping.gain_threshold)

    def aggregate(
        self,
        *,
        contexts: dict[Hashable, pd.DataFrame],
        effect: Any,
        parents: tuple,
        score_ctx: Callable[[pd.DataFrame], float],
    ) -> AggregationResult:
        # effect/parents are unused for LINC (scoring is encapsulated by score_ctx)
        ctx_ids = list(contexts.keys())
        n = len(ctx_ids)
        if n == 0:
            return AggregationResult(total=0.0, diagnostics={})

        # per-context scores
        ctx_scores: dict[Hashable, float] = {c: float(score_ctx(contexts[c])) for c in ctx_ids}

        if n == 1:
            return AggregationResult(
                total=float(ctx_scores[ctx_ids[0]]),
                diagnostics={"groups": [frozenset([ctx_ids[0]])], "ctx_scores": ctx_scores},
            )

        if self.grouping.method == "agglomerative":
            total, diag = self._agglomerative(ctx_ids, contexts, score_ctx)
            return AggregationResult(total=float(total), diagnostics=diag)

        # components method
        gain = np.zeros((n, n), dtype=float)
        edges: list[tuple[Hashable, Hashable]] = []

        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = ctx_ids[i], ctx_ids[j]
                pooled = pd.concat([contexts[ci], contexts[cj]], axis=0, ignore_index=True)
                pooled_score = float(score_ctx(pooled))
                g = float(self.transition_gain(ctx_scores[ci] + ctx_scores[cj], pooled_score))
                gain[i, j] = gain[j, i] = g
                if self.score_significant(g):
                    edges.append((ci, cj))

        self.last_gain_matrix = gain
        self.last_gain_contexts = tuple(ctx_ids)

        components = union_find_components(ctx_ids, edges)

        total = 0.0
        for comp in components:
            pooled = pd.concat([contexts[c] for c in comp], axis=0, ignore_index=True)
            total += float(score_ctx(pooled))

        diag = {
            "method": "components",
            "gain_matrix": gain,
            "gain_contexts": tuple(ctx_ids),
            "edges": edges,
            "groups": [frozenset(c) for c in components],
            "ctx_scores": ctx_scores,
        }
        return AggregationResult(total=float(total), diagnostics=diag)

    def _agglomerative(self, ctx_ids, contexts, score_ctx):
        groups = [frozenset([c]) for c in ctx_ids]

        def group_score(g):
            pooled = pd.concat([contexts[c] for c in g], axis=0, ignore_index=True)
            return float(score_ctx(pooled))

        scores = {g: group_score(g) for g in groups}

        while True:
            best_gain = float("-inf")
            best_pair = None
            best_score = None

            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    gi, gj = groups[i], groups[j]
                    merged = gi | gj
                    s_merged = group_score(merged)
                    g = float(self.transition_gain(scores[gi] + scores[gj], s_merged))
                    if g > best_gain:
                        best_gain = g
                        best_pair = (i, j, merged)
                        best_score = s_merged

            if best_pair is None or not self.score_significant(best_gain):
                break

            i, j, merged = best_pair
            gi, gj = groups[i], groups[j]

            groups = [g for k, g in enumerate(groups) if k not in (i, j)]
            groups.append(merged)

            scores.pop(gi)
            scores.pop(gj)
            scores[merged] = float(best_score)

        return float(sum(scores.values())), {
            "method": "agglomerative",
            "groups": list(scores.keys()),
            "scores": dict(scores),
        }
