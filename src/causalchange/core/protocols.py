from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, Protocol

import networkx as nx
import pandas as pd

from causalchange.core.results import GraphSearchResult, MultiContextResult, SCMClusteringResult
from causalchange.domain.tabular import TabularAllowedEdge, TabularNode, TabularScoreFunction
from causalchange.domain.temporal import TemporalAllowedEdge, TemporalNode, TemporalScoreFunction, TimeGrid


class TabularDomainProtocol(Protocol):
    def prepare_X(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def nodes(self, X0: pd.DataFrame) -> list[Any]: ...
    def candidates(self, X0: pd.DataFrame) -> list[Any]: ...
    def allowed_edge(self, cause: Any, effect: Any) -> bool: ...


class TemporalDomainProtocol(Protocol):
    def prepare_X(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def variables(self, X0: pd.DataFrame) -> list[str]: ...
    def allowed_edge(self, cause: TemporalNode, effect: TemporalNode) -> bool: ...


class ContextPreproc(Protocol):
    def make_contexts(self, X0: pd.DataFrame) -> dict[Hashable, pd.DataFrame]: ...
    def prepare_X(self, X: pd.DataFrame) -> pd.DataFrame: ...


class TabularScoringProtocol(Protocol):
    def fit(self, X0: pd.DataFrame) -> None: ...

    def local_score(
        self,
        df: pd.DataFrame,
        effect: Any,
        parents: tuple[Any, ...],
    ) -> float: ...

    @property
    def higher_is_better(self) -> bool: ...
    def transition_gain(self, old_score: float, new_score: float) -> float: ...
    def gain_is_better(self, a: float, b: float) -> bool: ...
    def score_significant(self, gain: float) -> bool: ...


class TemporalScoringProtocol(Protocol):
    def fit_panel(self, panel: TimeGrid) -> None: ...

    def set_time_windows(
        self,
        *,
        n_raw_samples: int,
        changepoints: list[int],
    ) -> None: ...

    def local_score(
        self,
        X: pd.DataFrame,
        effect: TemporalNode,
        parents: Sequence[TemporalNode],
        *,
        ret_full_result: bool = False,
        ret_residuals: bool = False,
    ) -> Any: ...

    def local_score_grid(
        self,
        *,
        panel: TimeGrid,
        effect: TemporalNode,
        parents: Sequence[TemporalNode],
        partitions: SCMClusteringResult,
    ) -> float: ...

    @property
    def higher_is_better(self) -> bool: ...

    def transition_gain(self, old_score: float, new_score: float) -> float: ...
    def gain_is_better(self, a: float, b: float) -> bool: ...
    def raw_score_is_better(self, a: float, b: float) -> bool: ...
    def score_significant(self, gain: float) -> bool: ...


class BaseContextCombination(Protocol):
    def combine_contexts(
        self,
        *,
        contexts: Mapping[Hashable, pd.DataFrame],
        effect: Any,
        parents: tuple[Any, ...],
        score_ctx: Callable[[pd.DataFrame], float],
    ) -> MultiContextResult: ...


class TabularSearchProtocol(Protocol):
    def run(
        self,
        *,
        nodes: Sequence[TabularNode],
        candidates: Sequence[TabularNode],
        allowed_edge: TabularAllowedEdge,
        score_fun: TabularScoreFunction,
    ) -> GraphSearchResult: ...


class TemporalSearchProtocol(Protocol):
    def run(
        self,
        *,
        variables: Sequence[str],
        tau_max: int,
        allowed_edge: TemporalAllowedEdge,
        score_fun: TemporalScoreFunction,
    ) -> GraphSearchResult: ...


class ChangepointDetectionProtocol(Protocol):
    changepoints_by_context_: dict[Any, list[int]] | None
    diagnostics_: dict[str, Any]

    def detect(
        self,
        X: pd.DataFrame | None = None,
        *,
        time_grid: TimeGrid | None = None,
        graph: nx.DiGraph | None = None,
        scorer: Any = None,
        variables: list[str] | None = None,
    ) -> list[int]: ...


class MechanismClusteringProtocol(Protocol):
    def fit_predict(
        self,
        X: pd.DataFrame | None = None,
        *,
        panel: TimeGrid | None = None,
        graph: Any = None,
        changepoints: list[int] | None = None,
        changepoints_by_context: dict[Any, list[int]] | None = None,
        scorer: Any = None,
    ) -> SCMClusteringResult: ...
