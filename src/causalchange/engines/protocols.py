from dataclasses import dataclass

import pandas as pd
from typing import Protocol, Any
from collections.abc import Callable, Hashable

from causalchange.discovery.graphsearch_tabular import DAGSearchResult


class Domain(Protocol):
    def prepare_X(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def nodes(self, X0: pd.DataFrame) -> list[Any]: ...
    def candidates(self, X0: pd.DataFrame) -> list[Any]: ...
    def allowed_edge(self, cause: Any, effect: Any) -> bool: ...


class ContextPreproc(Protocol):
    def make_contexts(self, X0: pd.DataFrame) -> dict[Hashable, pd.DataFrame]: ...
    def prepare_X(self, X: pd.DataFrame) -> pd.DataFrame: ...


class BaseScoring(Protocol):
    higher_is_better: bool

    def fit(self, X0: pd.DataFrame) -> None: ...
    def score_edge(self, df: pd.DataFrame, effect: Any, parents: tuple[Any, ...]) -> float: ...
    def transition_gain(self, old_score: float, new_score: float) -> float: ...
    def score_significant(self, gain: float) -> bool: ...
    def score_is_better(self, a: float, b: float) -> bool: ...


@dataclass(frozen=True)
class AggregationResult:
    total: float
    diagnostics: dict[str, Any]


class BaseAggregation(Protocol):
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
        score_fun: Callable[[Any, tuple[Any, ...]], float],
    ) -> DAGSearchResult: ...

