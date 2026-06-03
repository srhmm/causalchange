from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

# Format of our temporal nodes (node_nm, time_lag) where time_lag==0 if instantaneous, time_lag>0 ow
TemporalNode = tuple[str, int]
TemporalScoreFunction = Callable[[TemporalNode, tuple[TemporalNode, ...]], float]
TemporalAllowedEdge = Callable[[TemporalNode, TemporalNode], bool]


@dataclass
class TemporalDomain:
    """temporal preprocessing"""

    tau_max: int = 1
    allow_instantaneous: bool = True

    def __post_init__(self):
        if int(self.tau_max) <= 0:
            raise ValueError("tau_max must be positive.")
        self.tau_max = int(self.tau_max)

    def prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X.columns = [str(c) for c in X.columns]
        return X

    def variables(self, X: pd.DataFrame) -> list[str]:
        return [str(c) for c in X.columns]

    def nodes(self, X: pd.DataFrame) -> list[TemporalNode]:
        vars_ = self.variables(X)
        return [(v, lag) for lag in range(0, self.tau_max + 1) for v in vars_]

    def lagged_nodes(self, X: pd.DataFrame) -> list[TemporalNode]:
        vars_ = self.variables(X)
        return [(v, lag) for lag in range(1, self.tau_max + 1) for v in vars_]

    def current_nodes(self, X: pd.DataFrame) -> list[TemporalNode]:
        return [(v, 0) for v in self.variables(X)]

    def allowed_edge(self, cause: Any, effect: Any) -> bool:
        if not (isinstance(cause, tuple) and isinstance(effect, tuple) and len(cause) == 2 and len(effect) == 2):
            raise TypeError("TimeDomain expects nodes of form (variable, lag).")

        cause_var, cause_lag = cause
        effect_var, effect_lag = effect

        if effect_lag != 0:
            return False

        if cause_lag == 0:
            if not self.allow_instantaneous:
                return False
            return cause_var != effect_var

        return cause_lag > 0


@dataclass(frozen=True)
class TimeGrid:
    """the time datasets as a space-time grid
    For DataMode.TIME:
        datasets = {0: X}

    For DataMode.TIME_CONTEXTS:
        datasets = {context_id: X_context_without_context_col}
    """

    datasets: dict[Any, pd.DataFrame]
    variables: list[str]
    context_col: str | None = None

    @property
    def dataset_ids(self) -> list[Any]:
        return list(self.datasets.keys())

    @property
    def n_contexts(self) -> int:
        return len(self.datasets)

    def first_dataset(self) -> pd.DataFrame:
        return self.datasets[self.dataset_ids[0]]


def util_changepoints_to_intervals(
    n_samples: int,
    changepoints: list[int],
) -> list[tuple[int, int]]:
    """
    Convert changepoints into half-open intervals.

    Example:
        n_samples=100, changepoints=[30, 70]
        -> [(0, 30), (30, 70), (70, 100)]
    """
    cps = sorted(int(cp) for cp in changepoints)

    if any(cp <= 0 or cp >= n_samples for cp in cps):
        raise ValueError(f"changepoints must lie strictly inside [0, {n_samples}), got {changepoints}")

    if len(set(cps)) != len(cps):
        raise ValueError(f"changepoints must be unique, got {changepoints}")

    bounds = [0, *cps, int(n_samples)]
    return list(zip(bounds[:-1], bounds[1:], strict=False))
