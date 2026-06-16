from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from causalchange.config.causal_change_config import CausalChangeConfigBase
from causalchange.core.types import DataMode, MissingMode, ScoreType
from causalchange.scoring.regression import (
    fit_score_functional_model,
    fit_score_gam,
    fit_score_gp,
    fit_score_krr,
    fit_score_ln,
    fit_score_rff,
    fit_score_spln,
)


class ScoreConventionProtocol(Protocol):
    @property
    def higher_is_better(self) -> bool: ...

    def transition_gain(self, old_score: float, new_score: float) -> float: ...

    def gain_is_better(self, a: float, b: float) -> bool: ...

    def raw_score_is_better(self, a: float, b: float) -> bool: ...

    def score_significant(self, gain: float, n: int) -> bool: ...


class BaseLocalScorer:  # ABC):
    """High-level scoring, shared logic for tabular and temporal."""

    def __init__(self, cfg: CausalChangeConfigBase):
        self.data_mode: DataMode = cfg.data_mode
        self.score_type: ScoreConventionProtocol = cast(ScoreConventionProtocol, cfg.score_type)
        self.missing_mode: MissingMode = cfg.missing_mode
        self.score_params: dict[str, Any] = dict(cfg.score_kwargs or {})
        self._global_n_samples: int | None = None

    @property
    def higher_is_better(self) -> bool:
        return bool(self.score_type.higher_is_better)

    def transition_gain(self, old_score: float, new_score: float) -> float:
        return float(self.score_type.transition_gain(old_score, new_score))

    def gain_is_better(self, a: float, b: float) -> bool:
        return bool(self.score_type.gain_is_better(a, b))

    def raw_score_is_better(self, a: float, b: float) -> bool:
        return bool(self.score_type.raw_score_is_better(a, b))

    def score_significant(self, gain: float) -> bool:
        n = self._require_global_n_samples("score_significant")
        return bool(self.score_type.score_significant(gain, n))

    def local_gain(
        self,
        df: pd.DataFrame,
        effect: Any,
        base_parents: Iterable[Any],
        candidate_parent: Any,
    ) -> float:
        """Score gain from adding candidate_parent to base_parents.

        Observed-data scorers use ordinary local-score differences.
        Missing-aware scorers should override this method.
        """

        if self.missing_mode == MissingMode.MISSING:
            raise NotImplementedError(
                f"{type(self).__name__} has missing_mode='missing'. "
                "Standalone local_score differences are not valid; override local_gain()."
            )

        base_parents_t = tuple(base_parents)
        full_parents_t = (*base_parents_t, candidate_parent)

        old_score = self.local_score(df, effect, base_parents_t)
        new_score = self.local_score(df, effect, full_parents_t)

        return self.transition_gain(old_score, new_score)

    def local_score(self, *args: Any, **kwargs: Any) -> float | Any:
        if self.missing_mode == MissingMode.MISSING:
            raise NotImplementedError(
                f"{type(self).__name__} has missing_mode='missing'. "
                "local_score() is not meaningful for paired missing-aware scoring."
            )

        raise NotImplementedError(f"{type(self).__name__} must implement local_score().")

    def _set_global_n_samples(self, n_samples: int) -> None:
        self._global_n_samples = int(n_samples)

    def _require_global_n_samples(self, method_name: str) -> int:
        if self._global_n_samples is None:
            raise RuntimeError(f"Call fit(...) or fit_panel(...) before {method_name}().")

        return int(self._global_n_samples)

    @staticmethod
    def _stringify_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        return df


class SCMScore:
    """low-level scoring"""

    def __init__(
        self,
        data_mode: DataMode,
        score_type: ScoreType,
        **scoring_params: Any,
    ):
        self.data_mode = data_mode
        self.score_type = score_type
        self.scoring_params = scoring_params

        self.lg = scoring_params.get("lg", None)
        self.vb = scoring_params.get("vb", 0)
        self._info = (
            (lambda st: (self.lg.info(st) if self.lg is not None else print(st))) if self.vb > 0 else (lambda st: None)
        )

        # Memoization
        self.score_cache: dict[tuple[int, tuple[int, ...]], float] = {}
        self.res_cache: dict[tuple[int, tuple[int, ...]], dict] = {}

        self.X: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        """sets data"""
        self.X = np.asarray(X)
        self.score_cache.clear()
        self.res_cache.clear()

    def get_score_fun(self):  # within ScoreType?
        score_fun = (
            fit_score_krr
            if self.score_type == ScoreType.KRR
            else (
                fit_score_gp
                if self.score_type.value == ScoreType.GP.value
                else (
                    fit_score_rff
                    if self.score_type.value == ScoreType.FF.value
                    else (
                        fit_score_gam
                        if self.score_type == ScoreType.GAM
                        else (
                            fit_score_spln
                            if self.score_type == ScoreType.SPLINE
                            else (fit_score_ln if self.score_type == ScoreType.LIN else None)
                        )
                    )
                )
            )
        )
        if score_fun is None:
            raise ValueError(f"Unsupported score_type: {self.score_type}")
        return score_fun

    def local_score(
        self,
        j: int,
        pa: Sequence[int],
        ret_full_result: bool = True,
        ret_residuals: bool = False,
    ):
        """scores causal relationship pa->j"""
        if self.X is None:
            raise RuntimeError("Call fit(X) before local_score().")

        pa_key = tuple(sorted(int(p) for p in pa))
        key = (j, pa_key)

        if not ret_residuals and key in self.score_cache and key in self.res_cache:
            score = self.score_cache[key]
            res = self.res_cache[key]
            return (score, res) if ret_full_result else score

        score_fun = self.get_score_fun()
        score, res = fit_score_functional_model(
            self.X,
            pa=pa_key,
            target=j,
            score_fun=score_fun,
            ret_residuals=ret_residuals,
            **self.scoring_params,
        )

        if not ret_residuals:
            self.score_cache[key] = float(score)
            self.res_cache[key] = res

        return (float(score), res) if ret_full_result else float(score)
