from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from causalchange.config.cc_types import DataMode, GPType, ScoreType
from causalchange.discovery.scoring.fit import (
    fit_score_functional_model,
    fit_score_gam,
    fit_score_gp,
    fit_score_krr,
    fit_score_ln,
    fit_score_rff,
    fit_score_spln,
)


class EdgeScore:
    higher_is_better = True

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
                if self.score_type.value == GPType.EXACT.value
                else (
                    fit_score_rff
                    if self.score_type.value == GPType.FOURIER.value
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

    def score_edge(
        self,
        j: int,
        pa: Sequence[int],
        ret_full_result: bool = True,
        ret_residuals: bool = False,
    ):
        """scores causal relationship pa->j"""
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
