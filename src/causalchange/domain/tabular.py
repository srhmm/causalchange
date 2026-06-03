from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


class TabularDomain:
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
