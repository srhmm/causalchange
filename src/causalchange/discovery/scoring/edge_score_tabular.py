# causalchange/scoring/edge_score_tabular.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Mapping

import pandas as pd

from causalchange.config.config_types import DataMode, ScoreType
from causalchange.discovery.scoring.edge_score import EdgeScore


@dataclass(frozen=True)
class GainPolicy:
    """
    Small helper used by searchers/aggregators to interpret scores.
    """
    higher_is_better: bool
    gain_threshold: float = 0.0  # default: any positive gain

    def transition_gain(self, old_score: float, new_score: float) -> float:
        return (new_score - old_score) if self.higher_is_better else (old_score - new_score)

    def is_better(self, a: float, b: float) -> bool:
        # gains are compared in a consistent "higher is better" way
        return a > b

    def significant(self, gain: float) -> bool:
        return gain > self.gain_threshold


class EdgeScoreTabularScorer:
    """
    Tabular (IID/contexts) scorer based on EdgeScore.

    - binds X via EdgeScore.fit(X_np)
    - maps column names -> indices
    - exposes score_df(df, effect, parents)
    """

    def __init__(
        self,
        *,
        data_mode: DataMode,
        score_type: ScoreType,
        score_params: Mapping[str, Any] | None = None,
        higher_is_better: bool = False,
        gain_threshold: float = 0.0,
    ):
        if data_mode not in {DataMode.IID, DataMode.CONTEXTS, DataMode.TIME, DataMode.TIME_CONTEXTS}:
            raise ValueError(f"EdgeScoreTabularScorer does not support {data_mode=}")

        self.data_mode = data_mode
        self.score_type = score_type
        self.score_params = dict(score_params or {})
        self.policy = GainPolicy(higher_is_better=bool(higher_is_better), gain_threshold=float(gain_threshold))

        self._edges: EdgeScore | None = None
        self._col_index: dict[str, int] = {}

    @property
    def higher_is_better(self) -> bool:
        return self.policy.higher_is_better

    # --- lifecycle ---
    def fit(self, X0: pd.DataFrame) -> None:
        self.fit_df(X0)
    def fit_df(self, df: pd.DataFrame) -> None:
        """
        Bind this scorer to a particular dataset (df) and reset caches.
        Call this whenever you want to score edges w.r.t. a new df.
        """
        X_np = df.to_numpy(dtype=float)
        self._edges = EdgeScore(data_mode=self.data_mode, score_type=self.score_type, **self.score_params)
        # Step A you did: EdgeScore.fit binds X and clears caches
        self._edges.fit(X_np)
        self._col_index = {c: i for i, c in enumerate(df.columns)}

    # --- scoring ---

    def score_df(self, df: pd.DataFrame, effect: str, parents: Sequence[str]) -> float:
        """
        Score (parents -> effect) on the given dataframe.
        This is the replacement for _init_score + _score combined.

        For performance you typically call fit_df(df) once and then score_df on the same df many times.
        """
        if self._edges is None or not self._col_index:
            # convenience fallback: bind if not already bound
            self.fit_df(df)

        assert self._edges is not None
        j = self._col_index[effect]
        pa = [self._col_index[p] for p in parents]
        return float(self._edges.score_edge(j=j, pa=pa, ret_full_result=False))

    # --- policy helpers used by search/aggregation ---

    def transition_gain(self, old_score: float, new_score: float) -> float:
        return float(self.policy.transition_gain(old_score, new_score))

    def score_is_better(self, a: float, b: float) -> bool:
        return bool(self.policy.is_better(a, b))

    def score_significant(self, gain: float) -> bool:
        return bool(self.policy.significant(gain))
