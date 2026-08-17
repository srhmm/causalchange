from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

import pandas as pd

from causalchange.core.protocols import (
    BaseContextCombination,
    ContextPreproc,
    TabularDomainProtocol,
    TabularScoringProtocol,
    TabularSearchProtocol,
)
from causalchange.core.results import (
    CMMMixtureResult,
    ContextCombinationResult,
    LincMixtureResult,
    TabularResult,
)
from causalchange.core.types import DataMode, PostprocessingMode
from causalchange.engines.base import BaseDiscoveryEngine


class TabularDiscoveryEngine(BaseDiscoveryEngine[TabularDomainProtocol, TabularScoringProtocol, TabularSearchProtocol]):
    """shows lower-level control flow for tabular causal discovery"""

    def __init__(
        self,
        *,
        data_mode: DataMode,
        domain: TabularDomainProtocol,
        context_preproc: ContextPreproc,
        scoring: TabularScoringProtocol,
        context_comb: BaseContextCombination,
        search: TabularSearchProtocol,
        postprocessing_mode: PostprocessingMode = PostprocessingMode.SKIP,
    ):
        super().__init__(
            data_mode=data_mode,
            domain=domain,
            scoring=scoring,
            search=search,
            postprocessing_mode=postprocessing_mode,
        )

        self.domain = domain
        self.context_preproc = context_preproc
        self.context_comb = context_comb
        self.search = search

        self.X0_: pd.DataFrame | None = None
        self.contexts_: dict[Hashable, pd.DataFrame] | None = None
        self.last_context_combo_: ContextCombinationResult | None = None
        self._score_cache: dict[tuple[Any, tuple[Any, ...]], ContextCombinationResult] = {}

    def fit(self, X: pd.DataFrame) -> TabularDiscoveryEngine:
        self.contexts_ = self.context_preproc.make_contexts(X)
        X0 = self.context_preproc.prepare_X(X)
        X0 = self.domain.prepare_X(X0)
        self.scoring.fit(X0)
        self.X0_ = X0
        self._score_cache = {}
        return self

    def local_score(self, effect: Any, parents: Iterable[Any]) -> float:
        if self.contexts_ is None:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        parents_t: tuple[Any, ...] = tuple(sorted(parents, key=repr)) if parents is not None else tuple()
        cache_key = (effect, parents_t)

        if cache_key in self._score_cache:
            res = self._score_cache[cache_key]
            self.last_context_combo_ = res
            return float(res.total)

        def score_ctx(df: pd.DataFrame) -> float:
            return float(self.scoring.local_score(df, effect, parents_t))

        res = self.context_comb.combine_contexts(
            contexts=self.contexts_,
            effect=effect,
            parents=parents_t,
            score_ctx=score_ctx,
        )

        self._score_cache[cache_key] = res
        self.last_context_combo_ = res
        return float(res.total)

    def _run_discovery(self) -> TabularResult:
        if self.X0_ is None:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        nodes = self.domain.nodes(self.X0_)
        candidates = self.domain.candidates(self.X0_)

        graph_search_result = self.search.run(
            nodes=nodes,
            candidates=candidates,
            allowed_edge=self.domain.allowed_edge,
            score_fun=self.local_score,
        )

        cmm_mixture = self._extract_cmm_components(graph_search_result.graph)
        linc_mixture = self._extract_linc_components(graph_search_result.graph)

        return TabularResult(
            graph_search=graph_search_result,
            cmm_mixture=cmm_mixture,
            linc_mixture=linc_mixture,
            history=graph_search_result.history,
            diagnostics={
                "score_cache_size": len(self._score_cache),
                "has_cmm_mixture": cmm_mixture is not None,
                "has_linc_mixture": linc_mixture is not None,
            },
        )

    def _extract_cmm_components(self, graph) -> CMMMixtureResult | None:
        if self.X0_ is None:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        extractor = getattr(self.scoring, "fit_mixture_components", None)

        if extractor is None or not callable(extractor):
            return None

        return extractor(self.X0_, graph)

    def _extract_linc_components(self, graph) -> LincMixtureResult | None:
        extractor = getattr(self.context_comb, "fit_linc_components", None)
        if extractor is None or not callable(extractor):
            return None

        return extractor(graph)
