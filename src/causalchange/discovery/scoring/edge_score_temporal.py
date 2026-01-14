from __future__ import annotations

from typing import Sequence, Optional

import pandas as pd

from causalchange.config.cc_types import DataMode
from causalchange.config.cc_config import CausalChangeConfig

from causalchange.discovery.scoring.edge_score_tabular import EdgeScoreTabular

Node = tuple[str, int]  # (variable, lag)


class EdgeScoreTemporal:
    """scoring for temporal domain (both single and multi contexts)
    w lagged design matrix Z, everything else delegated to EdgeScoreTabular on Z"""

    def __init__(
        self,
        *,
        cfg: CausalChangeConfig,
    ):
        if cfg.data_mode not in {DataMode.TIME, DataMode.TIME_CONTEXTS}:
            raise ValueError(
                f"EdgeScoreTemporal expects temporal, got {cfg.data_mode=}"
            )
        if cfg.tau_max is None or cfg.tau_max <= 0:
            raise ValueError("provide (positive) tau_max (max time lag)")

        self.data_mode = cfg.data_mode
        self.score_type = cfg.score_type
        self.tau_max = cfg.tau_max
        self._tab = EdgeScoreTabular(cfg)

        self._node_to_col: dict[Node, str] = {}
        self._Z: Optional[pd.DataFrame] = None

    @property
    def higher_is_better(self) -> bool:
        return self._tab.higher_is_better

    def _ar_col(self, node: Node) -> str:
        v, lag = node
        return f"{v}_lag{lag}"

    def build_design(self, X: pd.DataFrame) -> pd.DataFrame:
        tau = self.tau_max
        cols: dict[str, pd.Series] = {}
        for v in X.columns:
            for lag in range(0, tau + 1):
                cols[self._ar_col((v, lag))] = X[v].shift(lag)

        Z = pd.DataFrame(cols)
        Z = Z.iloc[tau:].copy()
        Z.reset_index(drop=True, inplace=True)
        return Z

    def fit(self, X: pd.DataFrame) -> None:
        Z = self.build_design(X)
        self._node_to_col = {
            (v, lag): self._ar_col((v, lag))
            for v in X.columns
            for lag in range(0, self.tau_max + 1)
        }
        self._Z = Z
        self._tab.fit(Z)

    def score_edge(
        self, X: pd.DataFrame, effect: Node, parents: Sequence[Node]
    ) -> float:
        if self._Z is None or not self._node_to_col:
            self.fit(X)

        assert self._Z is not None
        eff = self._node_to_col[effect]
        par = [self._node_to_col[p] for p in parents]
        return float(self._tab.score_edge(self._Z, eff, par))

    def transition_gain(self, old_score: float, new_score: float) -> float:
        return self._tab.transition_gain(old_score, new_score)

    def score_is_better(self, a: float, b: float) -> bool:
        return self._tab.score_is_better(a, b)

    def score_significant(self, gain: float) -> bool:
        return self._tab.score_significant(gain)
