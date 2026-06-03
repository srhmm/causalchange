from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from causalchange.config.types import DataMode
from causalchange.discovery.graphsearch_tabular import DAGSearchResult
from causalchange.engines.protocols import Domain, ContextPreproc, BaseScoring, BaseAggregation, Search, \
    AggregationResult


class TabularDiscoveryEngine:
    """ shows lower-level control flow for tabular causal discovery. """
    def __init__(
        self,
        *,
        data_mode: DataMode,
        domain: Domain,
        context_preprocessing: ContextPreproc,
        scoring: BaseScoring,
        context_combination: BaseAggregation,
        search: Search,
    ):
        self.data_mode = data_mode

        self.domain = domain
        self.context_preproc = context_preprocessing
        self.scorer = scoring
        self.aggregator = context_combination
        self.search = search

        self.X0_: pd.DataFrame | None = None
        self.contexts_: dict[Hashable, pd.DataFrame] | None = None
        self.last_aggregation_: AggregationResult | None = None

        self._score_cache: dict[tuple[Any, tuple[Any, ...]], AggregationResult] = {}

    def fit(self, X: pd.DataFrame) -> TabularDiscoveryEngine:
        self.contexts_ = self.context_preproc.make_contexts(X)
        X0 = self.context_preproc.prepare_X(X)
        X0 = self.domain.prepare_X(X0)
        self.scorer.fit(X0)
        self.X0_ = X0
        self._score_cache = {}
        return self

    def score_edge(self, effect: Any, parents: Iterable[Any]) -> float:
        if self.contexts_ is None:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        parents_t: tuple[Any, ...] = tuple(sorted(parents, key=repr)) if parents is not None else tuple()
        cache_key = (effect, parents_t)

        if cache_key in self._score_cache:
            res = self._score_cache[cache_key]
            self.last_aggregation_ = res
            return float(res.total)

        def score_ctx(df: pd.DataFrame) -> float:
            return float(self.scorer.score_edge(df, effect, parents_t))

        res = self.aggregator.aggregate(
            contexts=self.contexts_,
            effect=effect,
            parents=parents_t,
            score_ctx=score_ctx,
        )

        self._score_cache[cache_key] = res
        self.last_aggregation_ = res
        return float(res.total)

    def discover(self) -> DAGSearchResult:
        if self.X0_ is None:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        nodes = self.domain.nodes(self.X0_)
        candidates = self.domain.candidates(self.X0_)

        result = self.search.run(
            nodes=nodes,
            candidates=candidates,
            allowed_edge=self.domain.allowed_edge,
            score_fun=self.score_edge,
        )

        result.edge_strengths = self._compute_edge_strengths(result.graph)
        result.diagnostics = {
            "data_mode": self.data_mode.value,
        }

        return result

    def _compute_edge_strengths(self, graph) -> dict[tuple[Any, Any], float]:
        strengths: dict[tuple[Any, Any], float] = {}

        for edge in graph.edges():
            parent, effect = edge

            parents = tuple(graph.predecessors(effect))
            score_with = self.score_edge(effect, parents)

            parents_without = tuple(p for p in parents if p != parent)
            score_without = self.score_edge(effect, parents_without)

            # MDL scores are lower-is-better, so positive means the edge helps.
            strengths[edge] = float(score_without - score_with)

        return strengths
