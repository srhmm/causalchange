# causalchange/scoring/edge_score_autoreg.py

from __future__ import annotations

from typing import Any, Sequence, Mapping, Optional

import pandas as pd

from causalchange.config.config_types import DataMode, ScoreType

from causalchange.discovery.scoring.edge_score_tabular import EdgeScoreTabularScorer

Node = tuple[str, int]  # (variable, lag)


class EdgeScoreTemporalScorer:
    """
    Autoregressive scorer for temporal nodes.
    Builds lagged design matrix Z, then delegates scoring to EdgeScoreTabularScorer on Z.
    """

    def __init__(
        self,
        *,
        data_mode: DataMode,
        score_type: ScoreType,
        tau_max: int,
        allow_instantaneous: bool = True,
        score_params: Mapping[str, Any] | None = None,
        higher_is_better: bool = False,
        gain_threshold: float = 0.0,
    ):
        if data_mode not in {DataMode.TIME, DataMode.TIME_CONTEXTS}:
            raise ValueError(f"EdgeScoreAutoRegressiveScorer expects temporal data_mode, got {data_mode=}")
        if tau_max <= 0:
            raise ValueError("tau_max must be positive.")

        self.data_mode = data_mode
        self.score_type = score_type
        self.tau_max = int(tau_max)
        self.allow_instantaneous = bool(allow_instantaneous)

        self._tab = EdgeScoreTabularScorer(
            data_mode=data_mode,
            score_type=score_type,
            score_params=score_params,
            higher_is_better=higher_is_better,
            gain_threshold=gain_threshold,
        )

        self._node_to_col: dict[Node, str] = {}
        self._Z: Optional[pd.DataFrame] = None

    @property
    def higher_is_better(self) -> bool:
        return self._tab.higher_is_better

    # --- design matrix helpers ---

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

    # --- lifecycle ---
    def fit(self, X0: pd.DataFrame) -> None:
        self.fit_df(X0)
    def fit_df(self, X: pd.DataFrame) -> None:
        """
        Bind scorer to a temporal dataframe X by creating AR design Z and binding the tabular scorer to Z.
        """
        Z = self.build_design(X)
        self._node_to_col = {
            (v, lag): self._ar_col((v, lag))
            for v in X.columns
            for lag in range(0, self.tau_max + 1)
        }
        self._Z = Z
        self._tab.fit_df(Z)

    # --- scoring ---

    def score_df(self, X: pd.DataFrame, effect: Node, parents: Sequence[Node]) -> float:
        """
        Score edge (parents -> effect) where effect/parents are temporal nodes (var, lag).
        """
        if self._Z is None or not self._node_to_col:
            self.fit_df(X)

        assert self._Z is not None
        eff = self._node_to_col[effect]
        par = [self._node_to_col[p] for p in parents]
        return float(self._tab.score_df(self._Z, eff, par))

    # --- policy passthrough ---

    def transition_gain(self, old_score: float, new_score: float) -> float:
        return self._tab.transition_gain(old_score, new_score)

    def score_is_better(self, a: float, b: float) -> bool:
        return self._tab.score_is_better(a, b)

    def score_significant(self, gain: float) -> bool:
        return self._tab.score_significant(gain)
