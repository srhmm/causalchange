from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Optional, Protocol, Any

import networkx as nx
import pandas as pd

from causalchange.config._cc_types import DataMode


class Domain(Protocol):
    def prepare_X(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def nodes(self, X0: pd.DataFrame) -> list[Any]: ...
    def candidates(self, X0: pd.DataFrame) -> list[Any]: ...
    def allowed_edge(self, cause: Any, effect: Any) -> bool: ...


class ContextProvider(Protocol):
    def make_contexts(self, X0: pd.DataFrame) -> dict[Hashable, pd.DataFrame]: ...


class BaseScorer(Protocol):
    higher_is_better: bool
    def fit(self, X0: pd.DataFrame) -> None: ...
    def score_df(self, df: pd.DataFrame, effect: Any, parents: tuple[Any, ...]) -> float: ...
    def transition_gain(self, old_score: float, new_score: float) -> float: ...
    def score_significant(self, gain: float) -> bool: ...
    def score_is_better(self, a: float, b: float) -> bool: ...


@dataclass(frozen=True)
class AggregationResult:
    total: float
    diagnostics: dict[str, Any]


class Aggregator(Protocol):
    def aggregate(
        self,
        *,
        contexts: dict[Hashable, pd.DataFrame],
        effect: Any,
        parents: tuple[Any, ...],
        score_ctx: Callable[[pd.DataFrame], float],
    ) -> AggregationResult: ...


class Search(Protocol):
    def run(
        self,
        *,
        nodes: list[Any],
        candidates: list[Any],
        allowed_edge: Callable[[Any, Any], bool],
        score_oracle: Callable[[Any, tuple[Any, ...]], float],
    ) -> nx.DiGraph: ...


class DiscoveryEngine:
    def __init__(
        self,
        *,
        data_mode: DataMode,
        domain: Domain,
        context_provider: ContextProvider,
        scorer: BaseScorer,
        aggregator: Aggregator,
        search: Search,
    ):
        self.data_mode = data_mode
        self.domain = domain
        self.context_provider = context_provider
        self.scorer = scorer
        self.aggregator = aggregator
        self.search = search

        self.X0_: Optional[pd.DataFrame] = None
        self.contexts_: Optional[dict[Hashable, pd.DataFrame]] = None
        self.last_aggregation_: Optional[AggregationResult] = None

    def fit(self, X: pd.DataFrame) -> "DiscoveryEngine":
        X0 = self.domain.prepare_X(X)

        # Scorer might use X0 for any global init (optional). It should not assume contexts.
        self.scorer.fit(X0)

        # Context provider splits X0 (or X, depending on your design) into per-context dfs
        self.contexts_ = self.context_provider.make_contexts(X0)
        self.X0_ = X0
        return self

    def score(self, effect: Any, parents: Iterable[Any]) -> float:
        if self.contexts_ is None:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        parents_t: tuple[Any, ...] = tuple(parents) if parents is not None else tuple()

        def score_ctx(df: pd.DataFrame) -> float:
            return float(self.scorer.score_df(df, effect, parents_t))

        res = self.aggregator.aggregate(
            contexts=self.contexts_,
            effect=effect,
            parents=parents_t,
            score_ctx=score_ctx,
        )
        self.last_aggregation_ = res
        return float(res.total)

    def discover(self) -> nx.DiGraph:
        if self.X0_ is None:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        nodes = self.domain.nodes(self.X0_)
        candidates = self.domain.candidates(self.X0_)

        return self.search.run(
            nodes=nodes,
            candidates=candidates,
            allowed_edge=self.domain.allowed_edge,
            score_oracle=self.score,
        )
