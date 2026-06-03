from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from causalchange.config.causal_change_config import CausalChangeConfigTabular
from causalchange.core.types import DataMode
from causalchange.scoring.base import SCMScore


@dataclass(frozen=True)
class GainPolicy:
    higher_is_better: bool
    gain_threshold_fn: Callable[[int], float] = lambda n: 0.0  # default any pos gain

    def transition_gain(self, old_score: float, new_score: float) -> float:
        return (new_score - old_score) if self.higher_is_better else (old_score - new_score)

    def is_better(self, a: float, b: float) -> bool:
        return a > b

    def is_significant(self, gain: float, n_samples: int) -> bool:
        return gain > float(self.gain_threshold_fn(n_samples))


class SCMScoreTabular:
    def __init__(self, cfg: CausalChangeConfigTabular):
        if cfg.data_mode not in (
            DataMode.IID,
            DataMode.CONTEXTS,
            DataMode.TIME,
            DataMode.TIME_CONTEXTS,
        ):
            raise ValueError(f"data_mode not valid or implemented {cfg.data_mode}")

        self.data_mode = cfg.data_mode
        self.score_type = cfg.score_type
        self.score_params = dict(cfg.score_kwargs or {})

        self.policy = GainPolicy(
            higher_is_better=bool(self.score_type.higher_is_better()),
            gain_threshold_fn=self.score_type.gain_threshold,
        )
        self._global_n_samples: int | None = None

        self._edges: SCMScore | None = None
        self._col_index: dict[str, int] = {}
        self._bound_key: tuple[int, tuple[str, ...], int] | None = None

    @property
    def higher_is_better(self) -> bool:
        return self.policy.higher_is_better

    def _df_key(self, df: pd.DataFrame) -> tuple[int, tuple[str, ...], int]:
        return (id(df), tuple(map(str, df.columns)), int(df.shape[0]))

    def _bind(self, df: pd.DataFrame) -> None:
        X_np = df.to_numpy(dtype=float)
        edges = SCMScore(data_mode=self.data_mode, score_type=self.score_type, **self.score_params)
        edges.fit(X_np)

        self._edges = edges
        self._col_index = {c: i for i, c in enumerate(df.columns)}
        self._bound_key = self._df_key(df)

    def _ensure_bound(self, df: pd.DataFrame) -> None:
        key = self._df_key(df)
        if self._edges is None or self._bound_key != key:
            self._bind(df)

    def fit(self, df: pd.DataFrame) -> None:
        self._global_n_samples = int(df.shape[0])
        self._bind(df)

    def score_edge(self, df: pd.DataFrame, effect: str, parents: Sequence[str]) -> float:
        self._ensure_bound(df)
        assert self._edges is not None

        j = self._col_index[effect]
        pa = [self._col_index[p] for p in parents]
        return float(self._edges.score_edge(j=j, pa=pa, ret_full_result=False))

    def transition_gain(self, old_score: float, new_score: float) -> float:
        return float(self.policy.transition_gain(old_score, new_score))

    def score_is_better(self, a: float, b: float) -> bool:
        return bool(self.policy.is_better(a, b))

    def score_significant(self, gain: float) -> bool:
        if self._global_n_samples is None:
            raise RuntimeError("Call fit(X0) before score_significant().")
        return bool(self.policy.is_significant(gain, self._global_n_samples))
