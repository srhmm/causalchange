from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

Node = tuple[str, int]


@dataclass
class TemporalDomain:
    """
    Temporal domain:
      - nodes are (var, lag) for lag=0..tau_max
      - candidates are lag-0 nodes only
      - allowed edges: only into lag-0 targets
    """

    tau_max: int = 1
    allow_instantaneous: bool = True

    def __post_init__(self):
        if int(self.tau_max) <= 0:
            raise ValueError("tau_max must be a positive integer.")
        self.tau_max = int(self.tau_max)

    def prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def nodes(self, X0: pd.DataFrame) -> list[Node]:
        vars_ = list(X0.columns)
        return [(v, lag) for lag in range(0, self.tau_max + 1) for v in vars_]

    def candidates(self, X0: pd.DataFrame) -> list[Node]:
        return [(v, 0) for v in list(X0.columns)]

    def parent_candidates(self, child: Node, remaining_lag0: Sequence[Node]) -> list[Node]:
        v_child, lag_child = child
        if lag_child != 0:
            raise ValueError("This design assumes only lag-0 nodes are scored as effects.")

        parents: list[Node] = []

        if self.allow_instantaneous:
            parents.extend([p for p in remaining_lag0 if p != child])

        # lagged parents are all variables at lags 1..tau_max
        vars_ = [v for (v, _) in remaining_lag0]
        for lag in range(1, self.tau_max + 1):
            parents.extend([(v, lag) for v in vars_])

        return parents

    def allowed_edge(self, u: Any, v: Any) -> bool:
        if not (isinstance(u, tuple) and isinstance(v, tuple) and len(u) == 2 and len(v) == 2):
            raise TypeError("TemporalDomain expects node tuples (var, lag).")

        _, lag_u = u
        _, lag_v = v

        if lag_v != 0:
            return False

        if not self.allow_instantaneous and lag_u == 0:
            return False
        return True
