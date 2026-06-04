from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from causalchange.config.causal_change_config import CausalChangeConfigBase
from causalchange.core.types import DataMode
from causalchange.scoring.base import BaseLocalScorer, SCMScore


class SCMScoreTabular(BaseLocalScorer):
    def __init__(self, cfg: CausalChangeConfigBase):
        super().__init__(cfg)

        if cfg.data_mode not in (
            DataMode.TABULAR,
            DataMode.TAB_CONTEXTS,
            DataMode.TIME,
            DataMode.TIME_CONTEXTS,
        ):
            raise ValueError(f"data_mode not valid or implemented: {cfg.data_mode}")

        self._edges: SCMScore | None = None
        self._col_index: dict[str, int] = {}
        self._bound_key: tuple[int, tuple[str, ...], int] | None = None

    def _df_key(self, df: pd.DataFrame) -> tuple[int, tuple[str, ...], int]:
        return (id(df), tuple(map(str, df.columns)), int(df.shape[0]))

    def _bind(self, df: pd.DataFrame) -> None:
        X_np = df.to_numpy(dtype=float)

        edges = SCMScore(
            data_mode=self.data_mode,
            score_type=self.score_type,
            **self.score_params,
        )
        edges.fit(X_np)

        self._edges = edges
        self._col_index = {str(c): i for i, c in enumerate(df.columns)}
        self._bound_key = self._df_key(df)

    def _ensure_bound(self, df: pd.DataFrame) -> None:
        key = self._df_key(df)
        if self._edges is None or self._bound_key != key:
            self._bind(df)

    def fit(self, df: pd.DataFrame) -> None:
        df = self._stringify_columns(df)

        self._set_global_n_samples(df.shape[0])
        self._bind(df)

    def local_score(
        self,
        df: pd.DataFrame,
        effect: str,
        parents: Sequence[str],
    ) -> float:
        self._ensure_bound(df)
        assert self._edges is not None

        effect = str(effect)
        parents = tuple(str(p) for p in parents)

        j = self._col_index[effect]
        pa = [self._col_index[p] for p in parents]

        return float(
            self._edges.local_score(
                j=j,
                pa=pa,
                ret_full_result=False,
            )
        )
