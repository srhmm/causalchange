from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional

import numpy as np
import pandas as pd

from pgmpy.causal_discovery.score_based import TOPIC
from pgmpy.estimators.ScoreCache import ScoreCache
from pgmpy.estimators.StructureScore import get_scoring_method


ScoreFn = Callable[[str, tuple[str, ...]], float]


class CHAINScoreMixin:
    context_col: str = "context"
    lambda_inv: float = 1.0
    mmd_max_samples: int = 200
    mmd_gamma: float | None = None
    mmd_compare_to: str = "pooled"

    _X_context: dict[Hashable, pd.DataFrame]
    context_score_fns: dict[Hashable, ScoreFn]
    _X_pooled: pd.DataFrame
    _rng: np.random.Generator
    _pooled_cache: dict[tuple[str, tuple[str, ...]], tuple[dict[Hashable, np.ndarray], np.ndarray]]
    _mmd_cache: dict[tuple[Any, ...], float]

    def __init__(
        self,
        *args,
        context_col: str = "context",
        lambda_inv: float = 1.0,
        mmd_max_samples: int = 200,
        mmd_gamma: float | None = None,
        mmd_compare_to: str = "pooled",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.context_col = context_col
        self.lambda_inv = float(lambda_inv)
        self.mmd_max_samples = int(mmd_max_samples)
        self.mmd_gamma = mmd_gamma
        self.mmd_compare_to = mmd_compare_to

        self._X_context = {}
        self.context_score_fns = {}
        self._rng = np.random.default_rng(0)
        self._pooled_cache = {}
        self._mmd_cache = {}

    def _parents_key(self, parents):
        return tuple(sorted(map(str, parents)))

    def _make_score_fn_for_df(self, df: pd.DataFrame) -> ScoreFn:
        _: Any
        score_cache: ScoreCache
        _, score_cache = get_scoring_method(self.scoring_method, df, True)  # type: ignore[attr-defined]
        return score_cache.local_score

    def _init_chain_score(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.context_col not in X.columns:
            raise ValueError(f"context_col '{self.context_col}' not found in X.columns")

        self._X_context = {}
        for context, g in X.groupby(self.context_col):
            self._X_context[context] = g.drop(columns=[self.context_col]).copy()

        self.context_score_fns = {}
        for context, Xc in self._X_context.items():
            self.context_score_fns[context] = self._make_score_fn_for_df(Xc)

        self._X_pooled = X.drop(columns=[self.context_col]).copy()
        self._pooled_cache = {}
        self._mmd_cache = {}

        return self._X_pooled

    def _score_simple(self, effect, parents) -> float:
        total = 0.0
        for score_fn in self.context_score_fns.values():
            total += float(score_fn(effect, parents))
        return total

    def _score(self, effect, parents) -> float:
        fit = 0.0
        for score_fn in self.context_score_fns.values():
            fit += float(score_fn(effect, parents))
        if self.lambda_inv <= 0 or len(self._X_context) <= 1:
            return float(fit)

        inv_pen = float(self._invariance_penalty(effect, parents))
        return float(fit - float(self.lambda_inv) * inv_pen)

    def _invariance_penalty(self, effect, parents) -> float:
        parents_t = tuple(parents)
        parents_key = tuple(sorted(str(p) for p in parents_t))

        mmd_key = (
            effect,
            parents_key,
            self.mmd_compare_to,
            int(self.mmd_max_samples),
            None if self.mmd_gamma is None else float(self.mmd_gamma),
        )
        if mmd_key in self._mmd_cache:
            return float(self._mmd_cache[mmd_key])

        residuals_by_c, pooled_resid = self._pooled_residuals_cached(effect, parents)

        pooled_resid_s = self._subsample_1d(pooled_resid, self.mmd_max_samples)
        if pooled_resid_s.size < 5 or len(residuals_by_c) <= 1:
            self._mmd_cache[mmd_key] = 0.0
            return 0.0

        if self.mmd_compare_to == "pairwise":
            cs = list(residuals_by_c.keys())
            pen = 0.0
            for i in range(len(cs)):
                ri = self._subsample_1d(residuals_by_c[cs[i]], self.mmd_max_samples)
                if ri.size < 5:
                    continue
                for j in range(i + 1, len(cs)):
                    rj = self._subsample_1d(residuals_by_c[cs[j]], self.mmd_max_samples)
                    if rj.size < 5:
                        continue
                    pen += self._mmd2_rbf(ri, rj, gamma=self.mmd_gamma)
            pen = float(pen)
        else:
            pen = 0.0
            for _, r in residuals_by_c.items():
                r_s = self._subsample_1d(r, self.mmd_max_samples)
                if r_s.size < 5:
                    continue
                pen += self._mmd2_rbf(r_s, pooled_resid_s, gamma=self.mmd_gamma)
            pen = float(pen)

        self._mmd_cache[mmd_key] = pen
        return float(pen)

    def _pooled_residuals_cached(self, effect, parents):
        parents_t = tuple(parents)
        parents_key = tuple(sorted(str(p) for p in parents_t))
        key = (str(effect), parents_key)

        if key in self._pooled_cache:
            return self._pooled_cache[key]

        residuals_by_c, pooled_resid = self._pooled_residuals(effect, parents)

        residuals_by_c = {c: self._normalize_residuals(r) for c, r in residuals_by_c.items()}
        pooled_resid = self._normalize_residuals(pooled_resid)

        self._pooled_cache[key] = (residuals_by_c, pooled_resid)
        return residuals_by_c, pooled_resid

    def _pooled_residuals(self, effect, parents):
        ys = []
        Xps = []
        by_context = {}

        for c, df in self._X_context.items():
            y = df[effect].to_numpy(dtype=float)
            if len(parents) == 0:
                Xp = None
            else:
                Xp = df[list(parents)].to_numpy(dtype=float)
            ys.append(y)
            if Xp is not None:
                Xps.append(Xp)
            by_context[c] = (Xp, y)

        y_all = np.concatenate(ys, axis=0)

        if len(parents) == 0:
            mu = float(np.mean(y_all))
            pooled_resid = y_all - mu
            resid_by_c = {c: y - mu for c, (_, y) in by_context.items()}
            return resid_by_c, pooled_resid

        X_all = np.concatenate(Xps, axis=0)
        X_all_i = np.column_stack([np.ones((X_all.shape[0], 1)), X_all])
        beta, *_ = np.linalg.lstsq(X_all_i, y_all, rcond=None)

        pooled_pred = X_all_i @ beta
        pooled_resid = y_all - pooled_pred

        resid_by_c = {}
        for c, (Xp, y) in by_context.items():
            Xp_i = np.column_stack([np.ones((Xp.shape[0], 1)), Xp])
            resid_by_c[c] = y - (Xp_i @ beta)

        return resid_by_c, pooled_resid

    def _subsample_1d(self, x: np.ndarray, max_n: int) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size <= max_n:
            return x
        idx = self._rng.choice(x.size, size=max_n, replace=False)
        return x[idx]

    def _mmd2_rbf(self, x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if x.shape[0] == 0 or y.shape[0] == 0:
            return 0.0

        if gamma is None:
            gamma = self._median_heuristic_gamma_1d(np.vstack([x, y]))

        Kxx = self._rbf_kernel(x, x, gamma)
        Kyy = self._rbf_kernel(y, y, gamma)
        Kxy = self._rbf_kernel(x, y, gamma)

        return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())

    def _rbf_kernel(self, a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
        aa = np.sum(a * a, axis=1, keepdims=True)
        bb = np.sum(b * b, axis=1, keepdims=True).T
        d2 = aa + bb - 2.0 * (a @ b.T)
        return np.exp(-gamma * d2)

    def _median_heuristic_gamma_1d(self, z: np.ndarray) -> float:
        n = z.shape[0]
        if n <= 2:
            return 1.0
        m = min(n, 300)
        idx = self._rng.choice(n, size=m, replace=False)
        zz = z[idx]
        d = np.abs(zz - zz.T)
        med = np.median(d[d > 0])
        if not np.isfinite(med) or med <= 0:
            return 1.0
        sigma = float(med)
        return 1.0 / (2.0 * sigma * sigma)

    def _normalize_residuals(self, r):
        r = np.asarray(r, dtype=float)
        mu = r.mean()
        sd = r.std()
        if sd < 1e-8:
            return r - mu
        return (r - mu) / sd

    def fit(self, X: pd.DataFrame, **kwargs):
        X_data = self._init_chain_score(X)

        init_score = getattr(super(), "_init_score", None)
        if callable(init_score):
            init_score(X_data)

        return super().fit(X_data, **kwargs)  # type: ignore[misc]

class CHAIN(CHAINScoreMixin, TOPIC):
    pass

