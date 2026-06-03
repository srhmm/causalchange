from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd

Node = tuple[str, int]


@dataclass
class TemporalDomain:
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

    def nodes(self, X: pd.DataFrame) -> list[Node]:
        vars_ = self.variables(X)
        return [(v, lag) for lag in range(0, self.tau_max + 1) for v in vars_]

    def lagged_nodes(self, X: pd.DataFrame) -> list[Node]:
        vars_ = self.variables(X)
        return [(v, lag) for lag in range(1, self.tau_max + 1) for v in vars_]

    def current_nodes(self, X: pd.DataFrame) -> list[Node]:
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
    """
    Collection of one or more aligned time-series datasets.

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
