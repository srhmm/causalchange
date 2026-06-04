from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

TabularNode = str
TabularEdge = tuple[TabularNode, TabularNode]
TabularScoreFunction = Callable[[TabularNode, tuple[TabularNode, ...]], float]  # EdgeScoreTabular.local_score()
TabularAllowedEdge = Callable[[TabularNode, TabularNode], bool]


class TabularDomain:
    """tabular "preprocessing" """

    def prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def nodes(self, X0: pd.DataFrame) -> list[str]:
        return list(X0.columns)

    def candidates(self, X0: pd.DataFrame) -> list[str]:
        return self.nodes(X0)

    def parent_candidates(self, child: str, remaining: Sequence[str]) -> list[str]:
        return [p for p in remaining if p != child]

    def allowed_edge(self, u: Any, v: Any) -> bool:
        return u != v
