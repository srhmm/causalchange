from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causalchange.config.causal_change_config import CausalChangeConfigTabular
from causalchange.core.types import MixedSCMType
from causalchange.scoring.base import BaseLocalScorer


@dataclass(frozen=True)
class _MixtureFit:
    score: float
    bic: float
    log_likelihood: float
    n_components: int
    n_parameters: int
    responsibilities: np.ndarray
    coefficients: np.ndarray
    variances: np.ndarray
    weights: np.ndarray


class SCMScoreCMM(BaseLocalScorer):
    """CMM local scorer,

    score(target | parents) is a latent mixture-regression BIC.
    """

    def __init__(self, cfg: CausalChangeConfigTabular):
        super().__init__(cfg)
        self.mix_type = cfg.mix_type

        if self.mix_type == MixedSCMType.SKIP:
            raise ValueError("SCMScoreCMM requires cfg.mix_type != MixedSCMType.SKIP.")

        self.k_max = int(self.score_params.get("k_max", 5))
        self.lambda_mix = float(self.score_params.get("lambda_mix", 0.0))
        self.hybrid_mixing = bool(self.score_params.get("hybrid_mixing", False))

        self.max_em_iter = int(self.score_params.get("max_em_iter", 100))
        self.tol = float(self.score_params.get("tol", 1e-5))
        self.ridge = float(self.score_params.get("ridge", 1e-8))
        self.min_variance = float(self.score_params.get("min_variance", 1e-8))
        self.n_init = int(self.score_params.get("n_init", 3))
        self.seed = int(getattr(cfg, "seed", 42))

        self.X_: np.ndarray | None = None
        self._col_index: dict[str, int] = {}
        self._bound_key: tuple[int, tuple[str, ...], int] | None = None
        self._score_cache: dict[tuple[int, tuple[int, ...]], float] = {}
        self._fit_cache: dict[tuple[int, tuple[int, ...]], _MixtureFit] = {}

    def _df_key(self, df: pd.DataFrame) -> tuple[int, tuple[str, ...], int]:
        return (id(df), tuple(map(str, df.columns)), int(df.shape[0]))

    def _bind(self, df: pd.DataFrame) -> None:
        df = self._stringify_columns(df)
        self.X_ = df.to_numpy(dtype=float)
        self._col_index = {str(c): i for i, c in enumerate(df.columns)}
        self._bound_key = self._df_key(df)
        self._score_cache.clear()
        self._fit_cache.clear()

    def _ensure_bound(self, df: pd.DataFrame) -> None:
        df = self._stringify_columns(df)
        key = self._df_key(df)
        if self.X_ is None or self._bound_key != key:
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
        assert self.X_ is not None

        effect = str(effect)
        parents = tuple(str(p) for p in parents)

        j = self._col_index[effect]
        pa = tuple(sorted(self._col_index[p] for p in parents))
        key = (j, pa)

        if key in self._score_cache:
            return self._score_cache[key]

        y = self.X_[:, j]
        X_pa = self.X_[:, list(pa)] if pa else np.empty((self.X_.shape[0], 0))

        fit = fit_cmm_local_score(
            X_pa=X_pa,
            y=y,
            mix_type=self.mix_type,
            k_max=self.k_max,
            lambda_mix=self.lambda_mix,
            hybrid_mixing=self.hybrid_mixing,
            max_em_iter=self.max_em_iter,
            tol=self.tol,
            ridge=self.ridge,
            min_variance=self.min_variance,
            n_init=self.n_init,
            seed=self.seed,
        )

        self._score_cache[key] = float(fit.score)
        self._fit_cache[key] = fit
        return float(fit.score)


def fit_cmm_local_score(
    *,
    X_pa: np.ndarray,
    y: np.ndarray,
    mix_type: MixedSCMType,
    k_max: int,
    lambda_mix: float,
    hybrid_mixing: bool,
    max_em_iter: int,
    tol: float,
    ridge: float,
    min_variance: float,
    n_init: int,
    seed: int,
) -> _MixtureFit:
    degree = _degree_from_mix_type(mix_type)
    Phi = _polynomial_design(X_pa, degree=degree)

    best: _MixtureFit | None = None
    max_k = max(1, int(k_max))

    for k in range(1, max_k + 1):
        fit = _fit_mixture_regression_em(
            Phi=Phi,
            y=np.asarray(y, dtype=float),
            n_components=k,
            max_em_iter=max_em_iter,
            tol=tol,
            ridge=ridge,
            min_variance=min_variance,
            n_init=n_init,
            seed=seed,
            lambda_mix=lambda_mix,
            hybrid_mixing=hybrid_mixing,
        )

        if best is None or fit.score < best.score:
            best = fit

    assert best is not None
    return best


def _degree_from_mix_type(mix_type: MixedSCMType) -> int:
    if mix_type == MixedSCMType.LIN:
        return 1
    if mix_type == MixedSCMType.QUADRATIC:
        return 2
    if mix_type == MixedSCMType.CUBIC:
        return 3
    if mix_type in {MixedSCMType.N_SPLINE, MixedSCMType.B_SPLINE}:
        raise NotImplementedError(
            f"CMM mix_type={mix_type.value!r} is not implemented yet. " "Use 'lin', 'quadratic', or 'cubic' first."
        )
    raise ValueError(f"Unsupported CMM mix_type={mix_type!r}.")


def _polynomial_design(X: np.ndarray, *, degree: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    cols = [np.ones((n, 1), dtype=float)]

    if X.shape[1] > 0:
        for d in range(1, degree + 1):
            cols.append(X**d)

    return np.hstack(cols)


def _fit_mixture_regression_em(
    *,
    Phi: np.ndarray,
    y: np.ndarray,
    n_components: int,
    max_em_iter: int,
    tol: float,
    ridge: float,
    min_variance: float,
    n_init: int,
    seed: int,
    lambda_mix: float,
    hybrid_mixing: bool,
) -> _MixtureFit:
    rng = np.random.default_rng(seed)
    best: _MixtureFit | None = None

    for init_idx in range(max(1, n_init)):
        resp = _initial_responsibilities(y, n_components, rng, init_idx)

        prev_ll = -np.inf
        coefs = np.zeros((n_components, Phi.shape[1]), dtype=float)
        variances = np.ones(n_components, dtype=float)
        weights = np.full(n_components, 1.0 / n_components, dtype=float)

        for _ in range(max_em_iter):
            weights, coefs, variances = _m_step(
                Phi=Phi,
                y=y,
                resp=resp,
                ridge=ridge,
                min_variance=min_variance,
            )
            resp, ll = _e_step(
                Phi=Phi,
                y=y,
                weights=weights,
                coefs=coefs,
                variances=variances,
            )

            if abs(ll - prev_ll) < tol:
                break

            prev_ll = ll

        n = len(y)
        p = Phi.shape[1]
        n_params = n_components * p + n_components + (n_components - 1)
        bic = -2.0 * prev_ll + n_params * np.log(max(n, 2))

        entropy = _responsibility_entropy(resp)
        score = bic + lambda_mix * entropy if hybrid_mixing else bic

        fit = _MixtureFit(
            score=float(score),
            bic=float(bic),
            log_likelihood=float(prev_ll),
            n_components=int(n_components),
            n_parameters=int(n_params),
            responsibilities=resp,
            coefficients=coefs,
            variances=variances,
            weights=weights,
        )

        if best is None or fit.score < best.score:
            best = fit

    assert best is not None
    return best


def _initial_responsibilities(
    y: np.ndarray,
    n_components: int,
    rng: np.random.Generator,
    init_idx: int,
) -> np.ndarray:
    n = len(y)

    if n_components == 1:
        return np.ones((n, 1), dtype=float)

    resp = np.zeros((n, n_components), dtype=float)

    if init_idx == 0:
        order = np.argsort(y)
        chunks = np.array_split(order, n_components)
        for k, idx in enumerate(chunks):
            resp[idx, k] = 1.0
    else:
        labels = rng.integers(0, n_components, size=n)
        resp[np.arange(n), labels] = 1.0

    empty = np.where(resp.sum(axis=0) == 0)[0]
    for k in empty:
        resp[rng.integers(0, n), :] = 0.0
        resp[rng.integers(0, n), k] = 1.0

    row_sum = resp.sum(axis=1, keepdims=True)
    return resp / np.maximum(row_sum, 1.0)


def _m_step(
    *,
    Phi: np.ndarray,
    y: np.ndarray,
    resp: np.ndarray,
    ridge: float,
    min_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, p = Phi.shape
    k_max = resp.shape[1]

    nk = resp.sum(axis=0)
    weights = nk / max(n, 1)

    coefs = np.zeros((k_max, p), dtype=float)
    variances = np.full(k_max, np.var(y) + min_variance, dtype=float)

    eye = np.eye(p, dtype=float)

    for k in range(k_max):
        w = resp[:, k]
        total_w = float(w.sum())

        if total_w <= 1e-12:
            weights[k] = 1e-12
            continue

        Phi_w = Phi * w[:, None]
        lhs = Phi.T @ Phi_w + ridge * eye
        rhs = Phi.T @ (w * y)

        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        resid = y - Phi @ beta
        var = float(np.sum(w * resid**2) / total_w)

        coefs[k] = beta
        variances[k] = max(var, min_variance)

    weights = weights / weights.sum()
    return weights, coefs, variances


def _e_step(
    *,
    Phi: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    coefs: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, float]:
    n = len(y)
    k_max = len(weights)

    log_prob = np.empty((n, k_max), dtype=float)

    for k in range(k_max):
        mean = Phi @ coefs[k]
        var = max(float(variances[k]), 1e-12)
        log_prob[:, k] = (
            np.log(max(float(weights[k]), 1e-12)) - 0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((y - mean) ** 2) / var
        )

    log_norm = _logsumexp(log_prob, axis=1)
    resp = np.exp(log_prob - log_norm[:, None])
    ll = float(np.sum(log_norm))

    return resp, ll


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)), axis=axis)


def _responsibility_entropy(resp: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(resp, eps, 1.0)
    return float(-np.sum(p * np.log(p)))
