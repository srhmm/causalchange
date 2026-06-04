from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, TypeVar

import pandas as pd

from causalchange.core.results import CausalChangeResult
from causalchange.core.types import DataMode, PostprocessingMode
from causalchange.posthoc.edge_strengths import compute_postprocessing_result


class EngineScoringProtocol(Protocol):
    def transition_gain(self, old_score: float, new_score: float) -> float: ...


DomainT = TypeVar("DomainT")
ScoringT = TypeVar("ScoringT", bound=EngineScoringProtocol)
SearchT = TypeVar("SearchT")


class BaseDiscoveryEngine(ABC, Generic[DomainT, ScoringT, SearchT]):
    """common control flow for discovery engines"""

    def __init__(
        self,
        *,
        data_mode: DataMode,
        domain: DomainT,
        scoring: ScoringT,
        search: SearchT,
        postprocessing_mode: PostprocessingMode = PostprocessingMode.SKIP,
    ):
        self.data_mode = data_mode
        self.domain = domain
        self.scoring = scoring
        self.search = search
        self.postprocessing_mode = postprocessing_mode

    @abstractmethod
    def fit(self, X: pd.DataFrame): ...

    @abstractmethod
    def local_score(self, effect: Any, parents) -> float:
        """score a target/effect with a proposed parent set"""
        ...

    @abstractmethod
    def _run_discovery(self) -> CausalChangeResult:
        """run causal discovery"""
        ...

    def discover(self) -> CausalChangeResult:
        """wrapper of run causal discovery and any postprocessing steps"""
        result = self._run_discovery()

        if self.postprocessing_mode != PostprocessingMode.SKIP:
            result.postprocessing = compute_postprocessing_result(
                result.graph,
                local_score=self.local_score,
                transition_gain=self.scoring.transition_gain,
            )

        return result
