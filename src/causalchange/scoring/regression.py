from __future__ import annotations

from collections.abc import Sequence
from math import log, log2
from typing import Any

import numpy as np
from numpy.linalg import inv
from scipy.special import comb
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from causalchange.core.types import MixedSCMType


def fit_regression_score(
    X: np.ndarray,
    pa: Sequence[int],
    target: int,
    score_fun,
    ret_residuals: bool = False,
    **scoring_params: Any,
):
    X = np.asarray(X)
    n = X.shape[0]

    pa = tuple(int(p) for p in pa)
    target = int(target)

    y = X[:, target]

    if len(pa) == 0:
        #  intercept only
        X_pa = np.ones((n, 1), dtype=float)
    else:
        X_pa = X[:, pa]

    if ret_residuals:
        model, L, resid = score_fun(X_pa, y, return_residuals=True, **scoring_params)
        return float(L), {"model": model, "residuals": resid}
    else:
        model, L = score_fun(X_pa, y, return_residuals=False, **scoring_params)
        return float(L), {"model": model}


def fit_score_gp(Xtr, ytr, return_residuals=False, **params):
    """
    GP refined-MDL score for a local causal mechanism.


        -log p(y | f, sigma^2) + ||f||_K^2
        + 0.5 log det(I + sigma^{-2} K)

    where f is the kernel-ridge/GP posterior mean under an RBF kernel.
    The returned score is in bits.
    """
    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float).reshape(-1)

    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(-1, 1)

    finite = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
    Xtr = Xtr[finite]
    ytr = ytr[finite]

    n = ytr.shape[0]
    min_n = int(params.get("min_n", 5))
    if n < min_n:
        score_bits = _null_gaussian_mdl_bits(ytr)

        ymu = float(np.mean(ytr)) if n else 0.0
        ysd = float(np.std(ytr)) if n else 1.0
        if ysd <= 1e-12:
            ysd = 1.0

        def predict(Xte, return_var=False):
            Xte = np.asarray(Xte, float)
            mean = np.full((Xte.shape[0],), ymu)
            if return_var:
                return mean, np.full_like(mean, ysd**2)
            return mean

        model = {
            "kind": "fallback_null_too_few_samples",
            "mdl_bits": float(score_bits),
            "predict": predict,
            "details": {"reason": f"too_few_samples: {n}"},
        }

        if return_residuals:
            resid = ytr - predict(Xtr)
            return model, float(score_bits), resid

        return model, float(score_bits)

    Xn, yn, scalers = _standardize(Xtr, ytr)
    n = Xn.shape[0]

    restarts = int(params.get("restarts", 10))
    low = float(params.get("bounds", {}).get("low", -5.0))
    high = float(params.get("bounds", {}).get("high", 5.0))
    refine = bool(params.get("refine", True))
    rng = np.random.default_rng(params.get("seed", None))

    base_jitter = float(params.get("base_jitter", 1e-8))
    min_noise_var = float(params.get("min_noise_var", 1e-8))
    norm_weight = float(params.get("rkhs_norm_weight", 1.0))

    # unnecessary
    use_bic = bool(params.get("bic_penalty", False))
    k_params = int(params.get("k_params", 3))

    cands = _random_restarts_bounds(3, low=low, high=high, rng=rng, n=restarts)
    cands += [
        np.array([0.0, 0.0, -2.0]),
        np.array([1.0, 0.0, -1.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -2.0, 0.0]),
        np.array([0.0, 2.0, 2.0]),
    ]

    best_theta = None
    best_score_nats = np.inf
    best_cache = None

    def eval_params(theta):
        log_ell, log_sf2, log_sn2 = theta
        log_sn2 = max(float(log_sn2), float(np.log(min_noise_var)))

        try:
            score_nats, cache = _gp_refined_mdl_terms(
                X=Xn,
                y=yn,
                log_ell=float(log_ell),
                log_sf2=float(log_sf2),
                log_sn2=log_sn2,
                base_jitter=base_jitter,
                norm_weight=norm_weight,
            )
        except np.linalg.LinAlgError:
            return np.inf, None

        if not np.isfinite(score_nats):
            return np.inf, None

        return float(score_nats), cache

    for theta in cands:
        score_nats, cache = eval_params(theta)
        if score_nats < best_score_nats:
            best_score_nats = score_nats
            best_theta = theta
            best_cache = cache

    if refine and best_theta is not None and np.isfinite(best_score_nats):
        for theta in _grid_around(best_theta, width=0.75, steps=3):
            score_nats, cache = eval_params(theta)
            if score_nats < best_score_nats:
                best_score_nats = score_nats
                best_theta = theta
                best_cache = cache

    if best_cache is None or not np.isfinite(best_score_nats):
        score_bits = _null_gaussian_mdl_bits(yn)
        Xmu, Xsd, ymu, ysd = scalers

        def predict(Xte, return_var=False):
            Xte = np.asarray(Xte, float)
            mean = np.full((Xte.shape[0],), ymu)
            if return_var:
                return mean, np.full_like(mean, ysd**2)
            return mean

        model = {
            "kind": "fallback_null",
            "mdl_bits": float(score_bits),
            "predict": predict,
            "scalers": {"Xmu": Xmu, "Xsd": Xsd, "ymu": ymu, "ysd": ysd},
        }

        if return_residuals:
            yhat = predict(Xtr)
            resid = ytr - yhat
            return model, float(score_bits), resid

        return model, float(score_bits)

    if use_bic:
        best_score_nats += 0.5 * k_params * np.log(max(n, 2))

    score_bits = best_score_nats / np.log(2.0)

    log_ell = best_cache["log_ell"]
    log_sf2 = best_cache["log_sf2"]
    log_sn2 = best_cache["log_sn2"]
    K_signal = best_cache["K_signal"]
    L = best_cache["L"]
    alpha = best_cache["alpha"]
    sigma2_eff = best_cache["sigma2_eff"]

    Xmu, Xsd, ymu, ysd = scalers

    def predict(Xte, return_var=False):
        Xte = np.asarray(Xte, float)
        if Xte.ndim == 1:
            Xte = Xte.reshape(-1, 1)

        Xte_n = (Xte - Xmu) / Xsd
        Kxs = _rbf_kernel(Xn, Xte_n, log_ell, log_sf2)
        mean_n = Kxs.T @ alpha
        mean = ymu + ysd * mean_n

        if not return_var:
            return mean

        Kxx = _rbf_kernel(Xte_n, Xte_n, log_ell, log_sf2)
        v = np.linalg.solve(L, Kxs)
        var_n = np.maximum(0.0, np.diag(Kxx) - np.sum(v**2, axis=0)) + sigma2_eff
        return mean, (ysd**2) * var_n

    yhat = predict(Xtr)
    resid = ytr - yhat

    model = {
        "kind": "gp_rbf_refined_mdl",
        "log_ell": float(log_ell),
        "log_sf2": float(log_sf2),
        "log_sn2": float(log_sn2),
        "sigma2_eff": float(sigma2_eff),
        "Xtr_std": Xn,
        "ytr_std": yn,
        "K_signal": K_signal,
        "L": L,
        "alpha": alpha,
        "predict": predict,
        "mdl_bits": float(score_bits),
        "score_nats": float(best_score_nats),
        "data_nll_nats": float(best_cache["data_nll_nats"]),
        "rkhs_norm": float(best_cache["rkhs_norm"]),
        "complexity_nats": float(best_cache["complexity_nats"]),
        "used_jitter": float(best_cache["used_jitter"]),
        "pointwise_error_bits": best_cache["pointwise_error_nats"] / np.log(2.0),
        "scalers": {"Xmu": Xmu, "Xsd": Xsd, "ymu": ymu, "ysd": ysd},
    }

    if return_residuals:
        return model, float(score_bits), resid

    return model, float(score_bits)


def _gp_refined_mdl_terms(
    *,
    X: np.ndarray,
    y: np.ndarray,
    log_ell: float,
    log_sf2: float,
    log_sn2: float,
    base_jitter: float,
    norm_weight: float,
):
    """
    Compute the refined-MDL GP terms in nats.

    Score =
        -log p(y | f, sigma^2)
        + norm_weight * ||f||_K^2
        + 0.5 log det(I + sigma^{-2} K)

    with f = K (K + sigma^2 I)^-1 y.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    n = y.shape[0]

    K_signal = _rbf_kernel(X, X, log_ell, log_sf2)
    sigma2 = float(np.exp(log_sn2))

    jitter = float(base_jitter)
    last_error = None

    for _ in range(8):
        sigma2_eff = sigma2 + jitter
        A = K_signal.copy()
        A.flat[:: n + 1] += sigma2_eff

        try:
            L = np.linalg.cholesky(A)
            alpha = _chol_solve(L, y)
            break
        except np.linalg.LinAlgError as exc:
            last_error = exc
            jitter *= 10.0
    else:
        raise last_error if last_error is not None else np.linalg.LinAlgError("Cholesky failed.")

    f_hat = K_signal @ alpha
    resid = y - f_hat

    data_nll_nats = 0.5 * (np.dot(resid, resid) / sigma2_eff + n * np.log(2.0 * np.pi * sigma2_eff))

    rkhs_norm = float(alpha @ K_signal @ alpha)

    # 0.5 log det(I + sigma^{-2} K)
    # Since A = K + sigma^2 I:
    # log det(I + K/sigma^2) = log det(A) - n log(sigma^2)
    logdet_A = 2.0 * np.sum(np.log(np.diag(L)))
    complexity_nats = 0.5 * (logdet_A - n * np.log(sigma2_eff))

    pointwise_error_nats = 0.5 * ((resid**2) / sigma2_eff + np.log(2.0 * np.pi * sigma2_eff))

    score_nats = data_nll_nats + norm_weight * rkhs_norm + complexity_nats

    return float(score_nats), {
        "log_ell": float(log_ell),
        "log_sf2": float(log_sf2),
        "log_sn2": float(log_sn2),
        "sigma2_eff": float(sigma2_eff),
        "K_signal": K_signal,
        "L": L,
        "alpha": alpha,
        "residuals_std": resid,
        "data_nll_nats": float(data_nll_nats),
        "rkhs_norm": float(rkhs_norm),
        "complexity_nats": float(complexity_nats),
        "pointwise_error_nats": pointwise_error_nats,
        "used_jitter": float(jitter),
    }


def fit_score_rff(Xtr, ytr, return_residuals=False, **params):
    """
    RFF approximation of the GP refined-MDL score.

    Approximates the RBF kernel by random Fourier features Phi and computes

        -log p(y | Phi w, sigma^2)
        + ||w||^2
        + 0.5 log det(I + sigma^{-2} Phi^T Phi)

    in bits.
    """
    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float).reshape(-1)

    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(-1, 1)

    finite = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
    Xtr = Xtr[finite]
    ytr = ytr[finite]

    n = ytr.shape[0]
    min_n = int(params.get("min_n", 5))
    if n < min_n:
        score_bits = _null_gaussian_mdl_bits(ytr)

        ymu = float(np.mean(ytr)) if n else 0.0
        ysd = float(np.std(ytr)) if n else 1.0
        if ysd <= 1e-12:
            ysd = 1.0

        def predict(Xte, return_var=False):
            Xte = np.asarray(Xte, float)
            mean = np.full((Xte.shape[0],), ymu)
            if return_var:
                return mean, np.full_like(mean, ysd**2)
            return mean

        model = {
            "kind": "fallback_null_rff_too_few_samples",
            "mdl_bits": float(score_bits),
            "predict": predict,
            "details": {"reason": f"too_few_samples: {n}"},
        }

        if return_residuals:
            resid = ytr - predict(Xtr)
            return model, float(score_bits), resid

        return model, float(score_bits)

    Xn, yn, scalers = _standardize(Xtr, ytr)
    n, d = Xn.shape

    D = int(params.get("D", 300))
    restarts = int(params.get("restarts", 10))
    low = float(params.get("bounds", {}).get("low", -5.0))
    high = float(params.get("bounds", {}).get("high", 5.0))
    refine = bool(params.get("refine", True))
    rng = np.random.default_rng(params.get("seed", None))

    base_jitter = float(params.get("base_jitter", 1e-8))
    min_noise_var = float(params.get("min_noise_var", 1e-8))
    norm_weight = float(params.get("rkhs_norm_weight", 1.0))

    # Optional small hyperparameter coding cost. Default False because the
    # refined-MDL score already contains the GP/RFF complexity term.
    use_bic = bool(params.get("bic_penalty", False))
    k_params = int(params.get("k_params", 3))

    # Fix random features across hyperparameter candidates. This makes the
    # score deterministic for a given seed and avoids comparing different
    # random bases during optimization.
    base_omegas = rng.standard_normal(size=(d, D))
    biases = rng.uniform(0.0, 2.0 * np.pi, size=(D,))

    cands = _random_restarts_bounds(3, low=low, high=high, rng=rng, n=restarts)
    cands += [
        np.array([0.0, 0.0, -2.0]),
        np.array([1.0, 0.0, -1.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -2.0, 0.0]),
        np.array([0.0, 2.0, 2.0]),
    ]

    best_theta = None
    best_score_nats = np.inf
    best_cache = None

    def eval_params(theta):
        log_ell, log_sf2, log_sn2 = theta
        log_sn2 = max(float(log_sn2), float(np.log(min_noise_var)))

        try:
            score_nats, cache = _rff_refined_mdl_terms(
                X=Xn,
                y=yn,
                base_omegas=base_omegas,
                biases=biases,
                log_ell=float(log_ell),
                log_sf2=float(log_sf2),
                log_sn2=log_sn2,
                base_jitter=base_jitter,
                norm_weight=norm_weight,
            )
        except np.linalg.LinAlgError:
            return np.inf, None

        if not np.isfinite(score_nats):
            return np.inf, None

        return float(score_nats), cache

    for theta in cands:
        score_nats, cache = eval_params(theta)
        if score_nats < best_score_nats:
            best_score_nats = score_nats
            best_theta = theta
            best_cache = cache

    if refine and best_theta is not None and np.isfinite(best_score_nats):
        for theta in _grid_around(best_theta, width=0.75, steps=3):
            score_nats, cache = eval_params(theta)
            if score_nats < best_score_nats:
                best_score_nats = score_nats
                best_theta = theta
                best_cache = cache

    if best_cache is None or not np.isfinite(best_score_nats):
        score_bits = _null_gaussian_mdl_bits(yn)
        Xmu, Xsd, ymu, ysd = scalers

        def predict(Xte, return_var=False):
            Xte = np.asarray(Xte, float)
            mean = np.full((Xte.shape[0],), ymu)
            if return_var:
                return mean, np.full_like(mean, ysd**2)
            return mean

        model = {
            "kind": "fallback_null_rff",
            "mdl_bits": float(score_bits),
            "predict": predict,
            "scalers": {"Xmu": Xmu, "Xsd": Xsd, "ymu": ymu, "ysd": ysd},
        }

        if return_residuals:
            yhat = predict(Xtr)
            resid = ytr - yhat
            return model, float(score_bits), resid

        return model, float(score_bits)

    if use_bic:
        best_score_nats += 0.5 * k_params * np.log(max(n, 2))

    score_bits = best_score_nats / np.log(2.0)

    log_ell = best_cache["log_ell"]
    log_sf2 = best_cache["log_sf2"]
    log_sn2 = best_cache["log_sn2"]
    sigma2_eff = best_cache["sigma2_eff"]
    w = best_cache["w"]
    A_chol = best_cache["A_chol"]

    Xmu, Xsd, ymu, ysd = scalers

    def predict(Xte, return_var=False):
        Xte = np.asarray(Xte, float)
        if Xte.ndim == 1:
            Xte = Xte.reshape(-1, 1)

        Xte_n = (Xte - Xmu) / Xsd
        Phi_te = _rff_features_from_base(
            X=Xte_n,
            base_omegas=base_omegas,
            biases=biases,
            log_ell=log_ell,
            log_sf2=log_sf2,
        )

        mean_n = Phi_te @ w
        mean = ymu + ysd * mean_n

        if not return_var:
            return mean

        # Bayesian linear model predictive observation variance.
        # A = Phi^T Phi + sigma^2 I
        tmp = np.linalg.solve(A_chol, Phi_te.T)
        var_n = sigma2_eff * (1.0 + np.sum(tmp**2, axis=0))
        return mean, (ysd**2) * var_n

    yhat = predict(Xtr)
    resid = ytr - yhat

    model = {
        "kind": "rff_refined_mdl",
        "D": int(D),
        "log_ell": float(log_ell),
        "log_sf2": float(log_sf2),
        "log_sn2": float(log_sn2),
        "sigma2_eff": float(sigma2_eff),
        "base_omegas": base_omegas,
        "biases": biases,
        "w": w,
        "A_chol": A_chol,
        "predict": predict,
        "mdl_bits": float(score_bits),
        "score_nats": float(best_score_nats),
        "data_nll_nats": float(best_cache["data_nll_nats"]),
        "weight_norm": float(best_cache["weight_norm"]),
        "complexity_nats": float(best_cache["complexity_nats"]),
        "used_jitter": float(best_cache["used_jitter"]),
        "pointwise_error_bits": best_cache["pointwise_error_nats"] / np.log(2.0),
        "scalers": {"Xmu": Xmu, "Xsd": Xsd, "ymu": ymu, "ysd": ysd},
    }

    if return_residuals:
        return model, float(score_bits), resid

    return model, float(score_bits)


def _rff_features_from_base(
    *,
    X: np.ndarray,
    base_omegas: np.ndarray,
    biases: np.ndarray,
    log_ell: float,
    log_sf2: float,
) -> np.ndarray:
    """
    Random Fourier features for an RBF kernel.

    With base_omegas ~ N(0, I), scaling by ell^{-1} gives
    omegas ~ N(0, ell^{-2} I). Multiplying by sqrt(sf2)
    approximates the signal variance.
    """
    X = np.asarray(X, float)
    ell = float(np.exp(log_ell))
    sf2 = float(np.exp(log_sf2))

    omegas = base_omegas / (ell + 1e-12)
    projection = X @ omegas + biases

    D = base_omegas.shape[1]
    return np.sqrt(2.0 * sf2 / D) * np.cos(projection)


def _rff_refined_mdl_terms(
    *,
    X: np.ndarray,
    y: np.ndarray,
    base_omegas: np.ndarray,
    biases: np.ndarray,
    log_ell: float,
    log_sf2: float,
    log_sn2: float,
    base_jitter: float,
    norm_weight: float,
):
    """
    RFF approximation of the GP refined-MDL terms in nats.

    Score =
        -log p(y | Phi w, sigma^2)
        + norm_weight * ||w||^2
        + 0.5 log det(I + sigma^{-2} Phi^T Phi)

    where
        w = (Phi^T Phi + sigma^2 I)^-1 Phi^T y.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)

    n = y.shape[0]
    D = base_omegas.shape[1]

    Phi = _rff_features_from_base(
        X=X,
        base_omegas=base_omegas,
        biases=biases,
        log_ell=log_ell,
        log_sf2=log_sf2,
    )

    sigma2 = float(np.exp(log_sn2))
    jitter = float(base_jitter)

    PtP = Phi.T @ Phi
    Pty = Phi.T @ y

    last_error = None
    for _ in range(8):
        sigma2_eff = sigma2 + jitter
        A = PtP.copy()
        A.flat[:: D + 1] += sigma2_eff

        try:
            A_chol = np.linalg.cholesky(A)
            w = _chol_solve(A_chol, Pty)
            break
        except np.linalg.LinAlgError as exc:
            last_error = exc
            jitter *= 10.0
    else:
        raise last_error if last_error is not None else np.linalg.LinAlgError("Cholesky failed.")

    yhat = Phi @ w
    resid = y - yhat

    data_nll_nats = 0.5 * (np.dot(resid, resid) / sigma2_eff + n * np.log(2.0 * np.pi * sigma2_eff))

    weight_norm = float(w @ w)

    # complexity = 0.5 log det(I + sigma^{-2} Phi^T Phi)
    # A = Phi^T Phi + sigma^2 I
    # log det(I + sigma^{-2} Phi^T Phi)
    # = log det(A) - D log(sigma^2)
    logdet_A = 2.0 * np.sum(np.log(np.diag(A_chol)))
    complexity_nats = 0.5 * (logdet_A - D * np.log(sigma2_eff))

    pointwise_error_nats = 0.5 * ((resid**2) / sigma2_eff + np.log(2.0 * np.pi * sigma2_eff))

    score_nats = data_nll_nats + norm_weight * weight_norm + complexity_nats

    return float(score_nats), {
        "log_ell": float(log_ell),
        "log_sf2": float(log_sf2),
        "log_sn2": float(log_sn2),
        "sigma2_eff": float(sigma2_eff),
        "Phi": Phi,
        "w": w,
        "A_chol": A_chol,
        "data_nll_nats": float(data_nll_nats),
        "weight_norm": float(weight_norm),
        "complexity_nats": float(complexity_nats),
        "pointwise_error_nats": pointwise_error_nats,
        "used_jitter": float(jitter),
    }


def _standardize(X, y, eps=1e-12):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    Xmu = X.mean(axis=0, keepdims=True)
    Xsd = X.std(axis=0, keepdims=True)
    Xsd = np.where(Xsd < eps, 1.0, Xsd)
    Xn = (X - Xmu) / Xsd
    ymu = y.mean()
    ysd = y.std()
    if ysd < eps:
        ysd = 1.0
    yn = (y - ymu) / ysd
    return Xn, yn, (Xmu, Xsd, ymu, ysd)


def _rbf_kernel(X, Z, log_ell, log_sf2):
    ell = np.exp(log_ell)
    sf2 = np.exp(log_sf2)
    X2 = np.sum(X**2, axis=1, keepdims=True)
    Z2 = np.sum(Z**2, axis=1, keepdims=True).T
    d2 = X2 + Z2 - 2.0 * X @ Z.T
    return sf2 * np.exp(-0.5 * d2 / (ell**2 + 1e-12))


def _chol_solve(L, b):
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)


def _neg_log_marginal_lik(y, K):
    n = y.shape[0]
    L = np.linalg.cholesky(K)
    alpha = _chol_solve(L, y)
    nll = 0.5 * (y @ alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2.0 * np.pi)
    if not np.isfinite(nll):
        nll = np.inf
    return float(nll), L, alpha


def _build_K_adaptive(X, log_ell, log_sf2, log_sn2, base_jitter=1e-6, max_tries=6):
    K = _rbf_kernel(X, X, log_ell, log_sf2)
    sn2 = np.exp(log_sn2)
    n = X.shape[0]
    jitter = base_jitter
    for _ in range(max_tries):
        K_try = K.copy()
        K_try.flat[:: n + 1] += sn2 + jitter
        try:
            _ = np.linalg.cholesky(K_try)
            return K_try, jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0

    K.flat[:: n + 1] += sn2 + jitter
    return K, jitter


def _random_restarts_bounds(d, low=-5.0, high=5.0, rng=None, n=12):
    rng = np.random.default_rng() if rng is None else rng
    return [rng.uniform(low, high, size=d) for _ in range(n)]


def _grid_around(best, width=0.75, steps=3):
    grids = []
    for i in range(len(best)):
        vs = best[i] + np.linspace(-width, width, steps)
        grids.append(vs)
    mesh = np.meshgrid(*grids, indexing="ij")
    cand = np.stack([m.reshape(-1) for m in mesh], axis=1)
    return [c for c in cand]


def _mdl_bits_from_nll(nll_nats, k_params, n):
    penalty = 0.5 * k_params * np.log(max(n, 2))
    return (nll_nats + penalty) / np.log(2.0)


def _null_gaussian_mdl_bits(y):
    n = len(y)
    if n == 0:
        return 1e9
    # mu = np.mean(y)
    var = np.var(y) + 1e-12
    nll = 0.5 * n * (np.log(2 * np.pi * var) + 1.0)
    k = 2
    return _mdl_bits_from_nll(nll, k, n)


def fit_score_ln(Xtr, ytr, return_residuals=False, **params):
    model_type = params.get("model_type", "ols").lower()
    alpha = float(params.get("alpha", 1.0))
    fit_intercept = bool(params.get("fit_intercept", True))
    param_penalty = params.get("param_penalty", "rissanen")

    Xtr = np.asarray(Xtr)
    ytr = np.asarray(ytr).ravel()
    n = Xtr.shape[0]

    if model_type == "ridge":
        base = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver="auto", random_state=None)
    else:
        base = LinearRegression(fit_intercept=fit_intercept)

    model = make_pipeline(StandardScaler(), base)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xtr)

    Phi = model[:-1].transform(Xtr)

    if isinstance(base, Ridge) and alpha > 0:
        k = _ridge_df_hat(Phi, alpha)
    else:
        k = _ols_df(Phi)

    nlml_bits, rss, sigma2 = _gaussian_nlml_bits(ytr, yhat)
    score_bits = nlml_bits + _penalty_bits(param_penalty, k, n)

    if return_residuals:
        resid = ytr - yhat
        return model, float(score_bits), resid

    return model, float(score_bits)


def fit_score_gam(Xtr, ytr, return_residuals=False, **params):
    n_knots = int(params.get("n_knots", 10))
    degree = int(params.get("degree", 3))
    include_bias = bool(params.get("include_bias", False))
    knots = params.get("knots", "quantile")
    extrapolation = params.get("extrapolation", "continue")
    model_type = params.get("model_type", "ridge").lower()
    alpha = float(params.get("alpha", 1.0))
    fit_intercept = bool(params.get("fit_intercept", True))
    param_penalty = params.get("param_penalty", "rissanen")

    Xtr = np.asarray(Xtr)
    ytr = np.asarray(ytr).ravel()
    n = Xtr.shape[0]

    spline = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        include_bias=include_bias,
        knots=knots,
        extrapolation=extrapolation,
    )
    if model_type == "ridge":
        reg = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    else:
        reg = LinearRegression(fit_intercept=fit_intercept)

    model = make_pipeline(StandardScaler(), spline, reg)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xtr)

    Phi = model[:-1].transform(Xtr)

    if isinstance(reg, Ridge) and alpha > 0:
        k = _ridge_df_hat(Phi, alpha)
    else:
        k = _ols_df(Phi)

    nlml_bits, rss, sigma2 = _gaussian_nlml_bits(ytr, yhat)
    score_bits = nlml_bits + _penalty_bits(param_penalty, k, n)

    if return_residuals:
        resid = ytr - yhat
        return model, float(score_bits), resid

    return model, float(score_bits)


def fit_score_krr(Xtr, ytr, return_residuals=False, **params):
    alpha = float(params.get("alpha", 1.0))
    gamma = params.get("gamma", None)
    param_penalty = params.get("param_penalty", "rissanen")

    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float).ravel()
    n = Xtr.shape[0]

    scaler = StandardScaler()
    Xtr_std = scaler.fit_transform(Xtr)

    krr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    krr.fit(Xtr_std, ytr)
    yhat = krr.predict(Xtr_std)

    k = n
    nlml_bits, rss, sigma2 = _gaussian_nlml_bits(ytr, yhat)
    score_bits = nlml_bits + _penalty_bits(param_penalty, k, n)

    def predict(Xte, return_var=False):
        Xte = np.asarray(Xte, float)
        Xte_std = scaler.transform(Xte)
        m = krr.predict(Xte_std)
        if return_var:
            return m, np.zeros_like(m)
        return m

    model = dict(
        kind="krr_rbf",
        mdl_bits=float(score_bits),
        predict=predict,
        scaler=scaler,
        alpha=float(alpha),
        gamma=gamma,
    )

    if return_residuals:
        resid = ytr - yhat
        return model, float(score_bits), resid

    return model, float(score_bits)


def _gaussian_nlml_bits(y_true, y_pred):
    n = y_true.shape[0]
    rss = np.sum((y_true - y_pred) ** 2)
    sigma2 = max(rss / n, 1e-30)
    nlml_nats = 0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)
    return nlml_nats / log(2), rss, sigma2


def _penalty_bits(param_penalty, k, n):
    if str(param_penalty).lower() in ("rissanen", "bic"):
        return (0.5 * k * np.log(n)) / log(2)
    return 0.0


def _ridge_df_hat(Phi, alpha):
    Phic = Phi - Phi.mean(axis=0, keepdims=True)
    G = Phic.T @ Phic
    A = G + alpha * np.eye(G.shape[0])
    H = Phic @ inv(A) @ Phic.T
    return float(np.trace(H)) + 1.0


def _ols_df(Phi):
    return float(Phi.shape[1] + 1)


class _SlopeBits:
    def logg(self, x):
        return 0.0 if x == 0 else np.log2(x)

    def logN(self, z):
        z = float(np.ceil(z))
        if z < 1:
            return 0.0
        log_star = self.logg(z)
        s = log_star
        while log_star > 0:
            log_star = self.logg(log_star)
            s += log_star
        return s + self.logg(2.865064)

    def model_score(self, coeff_vec):
        c = np.asarray(coeff_vec, float).ravel()
        c[~np.isfinite(c)] = 0.0
        s = 0.0
        for v in c:
            if abs(v) > 1e-12:
                c_abs = abs(v)
                c_dummy = c_abs
                precision = 1
                while c_dummy < 1000:
                    c_dummy *= 10.0
                    precision += 1
                s += self.logN(c_dummy) + self.logN(precision) + 1.0
        return s

    def gaussian_score_emp_sse(self, sse, n, min_diff):
        var = sse / max(n, 1)
        sigma = np.sqrt(max(var, 0.0))
        return self.gaussian_score_sse(sigma, sse, n, max(float(min_diff), 1e-12))

    def gaussian_score_sse(self, sigma, sse, n, resolution):
        sig2 = sigma * sigma
        if sse == 0.0 or sig2 == 0.0:
            return 0.0
        err = (
            (sse / (2.0 * sig2 * np.log(2.0))) + ((n / 2.0) * self.logg(2.0 * np.pi * sig2)) - n * self.logg(resolution)
        )
        return float(max(err, 0.0))


def _min_diff(y):
    y = np.asarray(y, float).ravel()
    y_sorted = np.sort(y)
    if y_sorted.size < 2:
        return 10.01
    diffs = np.diff(y_sorted)
    diffs = diffs[np.nonzero(diffs)]
    return float(np.min(diffs) if diffs.size else 10.01)


def _combinator(M, k):
    val = comb(M + k - 1, M, exact=False)
    return 0.0 if val <= 0 else np.log2(val)


def _aggregate_hinges(interactions, k, slope_bits, F):
    cost = 0.0
    for M in interactions:
        cost += slope_bits.logN(M) + _combinator(M, k) + M * np.log2(F)
    return float(cost)


def fit_score_spln(Xtr, ytr, return_residuals: bool = False, **params):
    X = np.asarray(Xtr, float)
    y = np.asarray(ytr, float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[finite]
    y = y[finite]

    min_n = max(5, int(params.get("min_n", 5)))
    if y.size < min_n:
        dummy = {
            "kind": "splines_sklearn",
            "model": None,
            "predict": (
                lambda Xte, return_var=False: (
                    (
                        np.full(len(Xte), np.nan),
                        np.full(len(Xte), np.nan),
                    )
                    if return_var
                    else np.full(len(Xte), np.nan)
                )
            ),
            "sse": float("inf"),
            "mdl_bits": float("inf"),
            "details": {"reason": f"too_few_finite_rows: {y.size}"},
        }
        if return_residuals:
            return dummy, float("inf"), np.array([])
        return dummy, float("inf")

    n, d = X.shape

    n_knots = int(params.get("n_knots", 10))
    degree = int(params.get("degree", 3))
    knots = params.get("knots", "quantile")
    extrapolation = params.get("extrapolation", "continue")
    include_bias = bool(params.get("include_bias", False))
    model_type = params.get("model_type", "ridge").lower()
    alpha = float(params.get("alpha", 1.0))
    globe_F = int(params.get("globe_F", 9))

    spline = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        include_bias=include_bias,
        knots=knots,
        extrapolation=extrapolation,
    )
    reg = Ridge(alpha=alpha) if model_type == "ridge" else LinearRegression()
    model = make_pipeline(StandardScaler(), spline, reg)
    # print(
    #    "X finite:",
    #    np.isfinite(X).all(),
    #    "n_nan:",
    #    np.isnan(X).sum(),
    #    "n_inf:",
    #    np.isinf(X).sum(),
    #    "col_nan_counts:",
    #    np.isnan(X).sum(axis=0),
    # )

    model.fit(X, y)

    yhat = model.predict(X)
    resid = y - yhat
    sse = float(np.dot(resid, resid))

    slope = _SlopeBits()
    k = np.array([d])
    dim = d
    rows = n
    mindiff = _min_diff(y)

    st = model.named_steps["splinetransformer"]
    knots_arr = getattr(st, "knots_", None)
    knots_flat = knots_arr.ravel() if knots_arr is not None else np.array([], dtype=float)
    coef = model.named_steps[list(model.named_steps.keys())[-1]].coef_
    coeffs_concat = np.concatenate([knots_flat, np.atleast_1d(coef).ravel()])

    model_bits = slope.model_score(coeffs_concat)

    Phi = model[:-1].transform(X)
    hinge_count = np.array([Phi.shape[1]], dtype=int)

    interactions = [1] * int(Phi.shape[1])

    base_cost = slope.model_score(k) + float(k[0]) * log2(dim if dim > 0 else 1)
    base_cost += slope.model_score(hinge_count)
    base_cost += _aggregate_hinges(interactions, int(k[0]), slope, globe_F)

    cost_bits = slope.gaussian_score_emp_sse(sse, rows, mindiff) + model_bits + base_cost

    def predict(Xte, return_var=False):
        ypred = model.predict(np.asarray(Xte, float))
        if return_var:
            sig2 = sse / max(rows, 1)
            return ypred, np.full(ypred.shape[0], sig2)
        return ypred

    out = {
        "kind": "splines_sklearn",
        "model": model,
        "predict": predict,
        "sse": sse,
        "mdl_bits": float(cost_bits),
        "details": {
            "coeff_model_bits": float(model_bits),
            "base_cost_bits": float(base_cost),
            "gaussian_bits": float(slope.gaussian_score_emp_sse(sse, rows, mindiff)),
            "hinge_count": hinge_count,
            "interactions": interactions,
            "n_knots": n_knots,
            "degree": degree,
            "alpha": alpha if model_type == "ridge" else 0.0,
        },
    }

    if return_residuals:
        return out, float(cost_bits), resid

    return out, float(cost_bits)


def to_params(params_r):
    params_np = np.array(params_r)
    D_plus1, K = params_np.shape
    param_list = []
    for k in range(K):
        beta_k = params_np[:-1, k]
        sigma_k = params_np[-1, k]
        param_list.append((beta_k, sigma_k))
    return param_list


def _require_rpy2_flexmix():
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import Formula, default_converter, numpy2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise ImportError(
            "requires 'rpy2'. "
            'Install it with `pip install "causalchange[cmm]"`. '
            "You also need R and the R package 'flexmix'."
        ) from exc

    try:
        flexmix = importr("flexmix")
    except Exception as exc:
        raise ImportError(
            "requires the R package 'flexmix'. " 'Install it in R with `install.packages("flexmix")`.'
        ) from exc

    return robjects, Formula, default_converter, numpy2ri, localconverter, flexmix


def fit_regression_mixture(
    mty: MixedSCMType,
    X,
    node_i,
    pa_i,
    range_k,
    resid,
    true_idl,
    lg=None,
    vb=0,
    degree=3,
):
    if mty == MixedSCMType.SKIP:
        raise ValueError("fit_conditional_mixture requires a concrete mix_type.")

    if mty == MixedSCMType.LIN:
        method = "lin"
    elif mty == MixedSCMType.QUADRATIC:
        method = "quad"
    elif mty == MixedSCMType.CUBIC:
        method = "cub"
    elif mty == MixedSCMType.N_SPLINE:
        method = "ns"
    elif mty == MixedSCMType.B_SPLINE:
        method = "bs"
    else:
        raise ValueError(f"Unsupported mix_type: {mty!r}")

    X = np.asarray(X, dtype=float)
    node_i = int(node_i)
    pa_i = [int(p) for p in pa_i]

    if not pa_i:
        y_only = X[:, node_i].reshape(-1, 1)

        best_bic = np.inf
        best_model = None

        for k in range_k:
            gm = GaussianMixture(n_components=int(k), random_state=42)
            gm.fit(y_only)

            bic_k = gm.bic(y_only)
            if bic_k < best_bic:
                best_bic = bic_k
                best_model = gm

        if best_model is None:
            raise RuntimeError("Failed to fit any marginal mixture model.")

        return {
            "bic": float(best_bic),
            "idl": best_model.predict(y_only),
            "pproba": best_model.predict_proba(y_only),
            "best_k": int(best_model.n_components),
        }

    if lg is not None and vb > 0:
        lg.info(f"Fitting mixture ({method})")

    (
        robjects,
        Formula,
        default_converter,
        numpy2ri,
        localconverter,
        flexmix,
    ) = _require_rpy2_flexmix()

    with localconverter(default_converter + numpy2ri.converter):
        y = X[:, node_i].reshape(-1, 1)
        X_pa = X[:, pa_i]

        data_np = np.hstack([y, X_pa])
        data_r = robjects.r.matrix(
            data_np,
            nrow=data_np.shape[0],
            ncol=data_np.shape[1],
        )

        robjects.r.assign("data_r", data_r)
        r_df = robjects.r["data.frame"](x=data_r)

        rhs_terms = []
        for i in range(X_pa.shape[1]):
            xi = f"x.{i + 2}"

            if method == "quad":
                rhs_terms.append(f"poly({xi}, 2)")
            elif method == "cub":
                rhs_terms.append(f"poly({xi}, 3)")
            elif method == "ns":
                rhs_terms.append(f"ns({xi}, df={degree})")
            elif method == "bs":
                rhs_terms.append(f"bs({xi}, df={degree})")
            else:
                rhs_terms.append(xi)

        formula_str = "x.1 ~ " + " + ".join(rhs_terms)
        formula = Formula(formula_str)

        if lg is not None and vb > 0:
            lg.info(f"Formula: {formula}")

        best_bic = np.inf
        best_model = None
        best_k = None

        for k in range_k:
            model = flexmix.flexmix(formula, data=r_df, k=int(k))
            bic = float(robjects.r["BIC"](model)[0])

            if vb:
                print(f"k={k}, BIC={bic}")

            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_k = int(k)

        if best_model is None or best_k is None:
            raise RuntimeError("Failed to fit any conditional mixture model.")

        post_probs = np.asarray(robjects.r["posterior"](best_model), dtype=float)
        hard_assign = post_probs.argmax(axis=1).astype(int)

    def post_entropy(p_proba, eps=1e-12):
        p_safe = np.clip(p_proba, eps, 1.0)
        return -np.sum(p_safe * np.log(p_safe), axis=1)

    return {
        "bic": float(best_bic),
        "idl": hard_assign,
        "pproba": post_probs,
        "entropy": post_entropy(post_probs),
        "best_k": int(best_k),
    }
