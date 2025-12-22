from __future__ import annotations
import numpy as np
from math import log
from pygam import GAM
from numpy.linalg import inv
from typing import Sequence, Any
from math import log2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LinearRegression, Ridge
from scipy.special import comb
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression

def fit_score_functional_model(
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
    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float).reshape(-1)

    if not np.all(np.isfinite(Xtr)) or not np.all(np.isfinite(ytr)):
        Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
        ytr = np.nan_to_num(ytr, nan=0.0, posinf=0.0, neginf=0.0)


    Xn, yn, scalers = _standardize(Xtr, ytr)
    n = Xn.shape[0]


    restarts = params.get("restarts", 10)
    low = params.get("bounds", {}).get("low", -5.0)
    high = params.get("bounds", {}).get("high", 5.0)
    refine = params.get("refine", True)
    rng = np.random.default_rng(params.get("seed", None))
    base_jitter = params.get("base_jitter", 1e-6)
    k_params = params.get("k_params", 3)
    use_bic = params.get("bic_penalty", False)


    cands = _random_restarts_bounds(3, low=low, high=high, rng=rng, n=restarts)
    cands += [
        np.array([0.0, 0.0, -2.0]),
        np.array([1.0, 0.0, -1.0]),
        np.array([-1.0, 0.0,  0.0]),
        np.array([0.0, -2.0, 0.0]),
        np.array([0.0,  2.0, 2.0]),
    ]

    best = None
    best_nll = np.inf
    best_cache = None
    best_jitter = base_jitter

    def eval_params(theta):
        log_ell, log_sf2, log_sn2 = theta
        log_sn2 = max(log_sn2, np.log(1e-6))
        K, used_jitter = _build_K_adaptive(Xn, log_ell, log_sf2, log_sn2, base_jitter=base_jitter)
        try:
            nll, L, alpha = _neg_log_marginal_lik(yn, K)
        except np.linalg.LinAlgError:
            return np.inf, None, used_jitter
        if not np.isfinite(nll):
            return np.inf, None, used_jitter
        return nll, (theta, K, L, alpha), used_jitter


    for th in cands:
        nll, cache, used_jit = eval_params(th)
        if nll < best_nll:
            best_nll, best, best_cache, best_jitter = nll, th, cache, used_jit


    if refine and best is not None and np.isfinite(best_nll):
        for th in _grid_around(best, width=0.75, steps=3):
            nll, cache, used_jit = eval_params(th)
            if nll < best_nll:
                best_nll, best, best_cache, best_jitter = nll, th, cache, used_jit


    if (best is None) or (not np.isfinite(best_nll)):
        score_bits = _null_gaussian_mdl_bits(yn) if use_bic else _null_gaussian_mdl_bits(yn)
        Xmu, Xsd, ymu, ysd = scalers

        def predict(Xte, return_var=False):
            Xte = np.asarray(Xte, float)
            m = np.full((Xte.shape[0],), ymu)
            if return_var:
                return m, np.full_like(m, ysd**2)
            return m

        model = dict(
            kind="fallback_null",
            mdl_bits=float(score_bits),
            predict=predict,
            scalers=dict(Xmu=Xmu, Xsd=Xsd, ymu=ymu, ysd=ysd),
        )

        if return_residuals:
            yhat = predict(Xtr)
            resid = ytr - yhat
            return model, float(score_bits), resid

        return model, float(score_bits)


    (log_ell, log_sf2, log_sn2), K, L, alpha = best_cache

    penalty = (0.5 * k_params * np.log(max(n, 2))) if use_bic else 0.0
    score_bits = (best_nll + penalty) / np.log(2.0)

    Xmu, Xsd, ymu, ysd = scalers

    def predict(Xte, return_var=False):
        Xte = np.asarray(Xte, float)
        Xte_n = (Xte - Xmu) / Xsd
        Kxs = _rbf_kernel(Xn, Xte_n, log_ell, log_sf2)
        mean_n = Kxs.T @ alpha
        if not return_var:
            return ymu + ysd * mean_n
        Kxx = _rbf_kernel(Xte_n, Xte_n, log_ell, log_sf2)
        v = np.linalg.solve(L, Kxs)
        var_n = np.maximum(0.0, np.diag(Kxx) - np.sum(v**2, axis=0)) + np.exp(log_sn2)
        return ymu + ysd * mean_n, (ysd**2) * var_n

    model = {
        "kind": "gp_rbf",
        "log_ell": float(log_ell),
        "log_sf2": float(log_sf2),
        "log_sn2": float(max(log_sn2, np.log(1e-6))),
        "Xtr_std": Xn,
        "ytr_std": yn,
        "L": L,
        "alpha": alpha,
        "predict": predict,
        "nll_nats": float(best_nll),
        "mdl_bits": float(score_bits),
        "used_jitter": float(best_jitter),
        "scalers": dict(Xmu=Xmu, Xsd=Xsd, ymu=ymu, ysd=ysd),
    }

    if return_residuals:
        yhat = predict(Xtr)
        resid = ytr - yhat
        return model, float(score_bits), resid

    return model, float(score_bits)

def fit_score_rff(Xtr, ytr, return_residuals=False, **params):
    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float).reshape(-1)
    if not np.all(np.isfinite(Xtr)) or not np.all(np.isfinite(ytr)):
        Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
        ytr = np.nan_to_num(ytr, nan=0.0, posinf=0.0, neginf=0.0)

    Xn, yn, scalers = _standardize(Xtr, ytr)
    n, d = Xn.shape

    restarts = int(params.get("restarts", 10))
    low = float(params.get("bounds", {}).get("low", -5.0))
    high = float(params.get("bounds", {}).get("high", 5.0))
    refine = bool(params.get("refine", True))
    rng = np.random.default_rng(params.get("seed", None))
    k_params = int(params.get("k_params", 3))
    use_bic = bool(params.get("bic_penalty", False))

    D = int(params.get("D", 300))
    omegas = rng.standard_normal(size=(d, D))
    biases = rng.uniform(0.0, 2.0*np.pi, size=(D,))

    def _rff_features(Xscaled):
        proj = Xscaled @ omegas
        return np.sqrt(2.0 / D) * np.cos(proj + biases)

    def _evidence_nlml_nats(log_ell, log_sf2, log_sn2):
        log_sn2 = max(log_sn2, np.log(1e-12))
        ell = np.exp(log_ell)
        sf2 = np.exp(log_sf2)
        sn2 = np.exp(log_sn2)

        Xs = Xn / (ell + 1e-12)
        Phi = _rff_features(Xs)
        PtP = Phi.T @ Phi
        b = Phi.T @ yn

        a = sf2 / sn2
        A = np.eye(D) + a * PtP
        try:
            L_A = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            return np.inf, None

        logdetS = n * np.log(sn2) + 2.0 * np.sum(np.log(np.diag(L_A)))
        tmp = np.linalg.solve(L_A, b)
        Ainv_b = np.linalg.solve(L_A.T, tmp)
        quad = (yn @ yn) / sn2 - (sf2 / (sn2**2)) * (b @ Ainv_b)

        nll = 0.5 * (logdetS + quad + n * np.log(2.0 * np.pi))
        cache = (ell, sf2, sn2, L_A, b, PtP)
        return nll, cache

    cands = _random_restarts_bounds(3, low=low, high=high, rng=rng, n=restarts)
    cands += [
        np.array([0.0, 0.0, -2.0]),
        np.array([1.0, 0.0, -1.0]),
        np.array([-1.0, 0.0,  0.0]),
        np.array([0.0, -2.0, 0.0]),
        np.array([0.0,  2.0, 2.0]),
    ]

    best = None
    best_nll = np.inf
    best_cache = None

    for th in cands:
        nll, cache = _evidence_nlml_nats(*th)
        if np.isfinite(nll) and nll < best_nll:
            best_nll, best, best_cache = nll, th, cache

    if refine and (best is not None) and np.isfinite(best_nll):
        for th in _grid_around(best, width=0.75, steps=3):
            nll, cache = _evidence_nlml_nats(*th)
            if np.isfinite(nll) and nll < best_nll:
                best_nll, best, best_cache = nll, th, cache

    if (best is None) or (not np.isfinite(best_nll)):
        score_bits = _null_gaussian_mdl_bits(yn)
        Xmu, Xsd, ymu, ysd = scalers

        def predict(Xte, return_var=False):
            Xte = np.asarray(Xte, float)
            m = np.full((Xte.shape[0],), ymu)
            if return_var:
                return m, np.full_like(m, ysd**2)
            return m

        model = dict(
            kind="fallback_null_rff",
            mdl_bits=float(score_bits),
            predict=predict,
            scalers=dict(Xmu=Xmu, Xsd=Xsd, ymu=ymu, ysd=ysd),
        )

        if return_residuals:
            yhat = predict(Xtr)
            resid = ytr - yhat
            return model, float(score_bits), resid

        return model, float(score_bits)

    (log_ell, log_sf2, log_sn2) = best
    ell, sf2, sn2, L_A, b_vec, PtP = best_cache

    penalty_nats = (0.5 * k_params * np.log(max(n, 2))) if use_bic else 0.0
    score_bits = (best_nll + penalty_nats) / np.log(2.0)

    Xmu, Xsd, ymu, ysd = scalers

    def predict(Xte, return_var=False):
        Xte = np.asarray(Xte, float)
        Xte_n = (Xte - Xmu) / Xsd
        Xte_s = Xte_n / (ell + 1e-12)
        Phi_star = np.sqrt(2.0 / D) * np.cos((Xte_s @ omegas) + biases)

        tmp = np.linalg.solve(L_A, b_vec)
        Ainv_b = np.linalg.solve(L_A.T, tmp)
        mu_w = (sf2 / sn2) * Ainv_b

        mean_n = Phi_star @ mu_w
        mean = ymu + ysd * mean_n
        if not return_var:
            return mean

        tmp2 = np.linalg.solve(L_A, Phi_star.T)
        quad = np.sum(tmp2**2, axis=0)
        var_n = sn2 + sf2 * quad
        return mean, (ysd**2) * var_n

    model = {
        "kind": "gp_rff",
        "log_ell": float(log_ell),
        "log_sf2": float(log_sf2),
        "log_sn2": float(max(log_sn2, np.log(1e-12))),
        "ell": float(ell),
        "sf2": float(sf2),
        "sn2": float(sn2),
        "Xtr_std": Xn,
        "ytr_std": yn,
        "predict": predict,
        "nll_nats": float(best_nll),
        "mdl_bits": float(score_bits),
        "rff": {"D": D, "omegas": omegas, "biases": biases, "PtP": PtP, "A_chol": L_A, "b": b_vec},
        "scalers": dict(Xmu=Xmu, Xsd=Xsd, ymu=ymu, ysd=ysd),
    }

    if return_residuals:
        yhat = predict(Xtr)
        resid = ytr - yhat
        return model, float(score_bits), resid

    return model, float(score_bits)


def _standardize(X, y, eps=1e-12):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    Xmu = X.mean(axis=0, keepdims=True)
    Xsd = X.std(axis=0, keepdims=True)
    Xsd = np.where(Xsd < eps, 1.0, Xsd)
    Xn = (X - Xmu) / Xsd
    ymu = y.mean()
    ysd = y.std()
    if ysd < eps:  # degenerate target: keep ysd=1 to avoid blowups
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
        K_try.flat[:: n + 1] += (sn2 + jitter)
        try:
            _ = np.linalg.cholesky(K_try)
            return K_try, jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0

    K.flat[:: n + 1] += (sn2 + jitter)
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
    mu = np.mean(y)
    var = np.var(y) + 1e-12
    nll = 0.5 * n * (np.log(2 * np.pi * var) + 1.0)
    k = 2
    return _mdl_bits_from_nll(nll, k, n)

def fit_score_gam_alt(Xtr, ytr):
    gam = GAM()
    gam.fit(Xtr, ytr)
    n_splines, order = 20, 3
    mse = np.mean((gam.predict(Xtr) - ytr) ** 2)
    n = Xtr.shape[0]
    p = Xtr.shape[1] * n_splines * order
    gam.mdl_lik_train = n * np.log(mse)
    gam.mdl_model_train = 2 * p
    gam.mdl_pen_train = 0
    gam.mdl_train = gam.mdl_lik_train + gam.mdl_model_train + gam.mdl_pen_train
    return gam, gam.mdl_train
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
        extrapolation=extrapolation
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

from sklearn.kernel_ridge import KernelRidge

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
    return float(np.trace(H)) + 1.0  # +1 intercept

def _ols_df(Phi):
    return float(Phi.shape[1] + 1)   # params + intercept


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
        err = (sse / (2.0 * sig2 * np.log(2.0))) + ((n / 2.0) * self.logg(2.0 * np.pi * sig2)) - n * self.logg(resolution)
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
            "predict": (lambda Xte, return_var=False: (
            np.full(len(Xte), np.nan), np.full(len(Xte), np.nan)) if return_var else np.full(len(Xte), np.nan)),
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
    print("X finite:", np.isfinite(X).all(),
          "n_nan:", np.isnan(X).sum(),
          "n_inf:", np.isinf(X).sum(),
          "col_nan_counts:", np.isnan(X).sum(axis=0))



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
    params_np = np.array(params_r)  # (D+1, K)
    D_plus1, K = params_np.shape
    param_list = []
    for k in range(K):
        beta_k = params_np[:-1, k]
        sigma_k = params_np[-1, k]
        param_list.append((beta_k, sigma_k))
    return param_list


def mix_regression_params_kn_assgn(X, y, idl):
    K = np.unique(idl).size
    beta_l, sigma_l = [], []

    for k in range(K):
        X_k, y_k = X[idl == k], y[idl == k]

        if len(y_k) == 0:
            beta_l.append(np.full(X.shape[1], np.nan))
            sigma_l.append(np.nan)
            continue

        model = LinearRegression(fit_intercept=False)  # todo intercept no right?
        model.fit(X_k, y_k)
        beta_k = model.coef_

        residuals = y_k - X_k @ beta_k
        sigma_k = np.sqrt(np.mean(residuals ** 2))

        beta_l.append(beta_k)
        sigma_l.append(sigma_k)

    return beta_l, sigma_l


def mix_regression_bic(X, y, idl, beta_l, sigma_l):
    N, D = X.shape
    K = len(beta_l)

    log_likelihood = 0.0
    mixture_counts = np.array([np.sum(idl == k) for k in range(K)])
    mixture_weights = mixture_counts / N

    for k in range(K):
        X_k = X[idl == k]
        y_k = y[idl == k]
        beta_k = beta_l[k]
        sigma_k = sigma_l[k]

        if len(y_k) == 0: continue

        residuals = y_k - X_k @ beta_k
        log_likelihood += np.sum(-0.5 * np.log(2 * np.pi * sigma_k ** 2) - 0.5 * (residuals ** 2) / sigma_k ** 2)
        log_likelihood += len(y_k) * np.log(
            mixture_weights[k] + 1e-12)
    num_params = K * D + K + (K - 1)

    bic = -2 * log_likelihood + num_params * np.log(N)
    return bic


def fit_conditional_mixture(mty: MixingType, **kwargs):
    assert mty.value != MixingType.SKIP.value

    if mty.value.startswith('mix'):
        method= 'quad' if mty.value=='mixQuad' else 'cub' if mty.value=='mixCub' else 'ns' if mty.value=='mixNS' else \
            'bs' if mty.value=='mixBS' else 'lin'
        return fit_functional_mixture(**kwargs, method=method)
    elif mty.value.startswith('resid'):
        return fit_resid_mixture(mty, **kwargs)
    elif mty.value.startswith('clus'):
        return fit_marginal_mixture(mty, **kwargs)
    else:
        raise ValueError(mty)


def _fit_best_mixture(X, range_k, true_idl, sim_score=adjusted_mutual_info_score, sim_min=-np.inf):
    best_ami = sim_min
    best_arg = None
    for mty in [MixingType.BASE_GMM, MixingType.BASE_KMEANS, MixingType.BASE_SPECTRAL, MixingType.BASE_DBSCAN]:
        idl, pproba, div = fit_mixture_model(mty, X, range_k, None)
        ami = sim_score(true_idl, idl)
        if ami > best_ami:
            best_ami = ami
            best_arg = idl, pproba, div
    res_dict = dict(
        bic=0,
        idl=best_arg[0],
        pproba=best_arg[1],
    )
    return res_dict


def fit_mixture_model(mty, X, range_k, true_idl=None, kchoice_score=silhouette_score, kchoice_threshold=0.5,
                      kchoice_min=-1):
    if mty == MixingType.BASE_RANDOM_SPLIT:
        assert true_idl is not None
        true_k = len(np.unique(true_idl))
        # sample random labels with true k
        rand_split = np.random.choice(true_k, size=len(true_idl))
        res_dict = dict( bic=0, idl=rand_split )
        return rand_split, None, dict()

    elif mty == MixingType._BASE_BEST:
        assert true_idl is not None
        return _fit_best_mixture(X, range_k, true_idl)

    elif mty in [MixingType.BASE_GMM, MixingType.BASE_GMM_GLOB]:
        mm = GaussianMixture
        best_bic, best_k, best_m = np.inf, 0, None
        for k in range_k:
            gm = mm(k)
            gm.fit(X)
            bic_k = gm.bic(X)
            if bic_k < best_bic: best_bic, best_k, best_m = bic_k, k, gm

        res_dict = dict( bic=best_bic, idl=best_m.predict(X), pproba = best_m.predict_proba(X))
        return res_dict

    elif mty == MixingType.BASE_DBSCAN:
        mm = DBSCAN().fit(X)
        res_dict = dict(idl=mm.labels_)
        return res_dict
    elif mty == MixingType.BASE_HDBSCAN:

        from sklearn.cluster import  HDBSCAN
        mm = HDBSCAN().fit(X)
        res_dict = dict(idl=mm.labels_)
        return res_dict
    else:
        model = KMeans if mty == MixingType.BASE_KMEANS \
            else SpectralClustering if mty == MixingType.BASE_SPECTRAL else None
        if model is None: raise ValueError(mty)
        best_s, best_k, best_idl = kchoice_min, 1, None
        for k in range_k:
            if k == 1: continue
            mm = model(n_clusters=k, random_state=42)
            idl = mm.fit_predict(X)
            s = kchoice_score(X, idl)
            if s > best_s: best_s, best_k, best_idl = s, k, idl
        if best_s < kchoice_threshold:  best_idl = model(n_clusters=1, random_state=42).fit_predict(X)
        res_dict = dict(idl=best_idl)
        return res_dict


def fit_marginal_mixture(mty, X, node_i, pa_i, range_k, resid, true_idl, **kwargs):
    X = np.hstack([X[:, pa_i], X[:, node_i].reshape(-1, 1)]) if len(pa_i) > 0 else X[:, node_i].reshape(-1, 1)
    return fit_mixture_model(mty, X, range_k, true_idl)


def fit_resid_mixture(mty, X, node_i, pa_i, range_k, resid, true_idl):
    return fit_mixture_model(mty, resid, range_k)


def fit_functional_mixture(
        X, node_i, pa_i, range_k, resid, true_idl,
        lg=None, vb=0, degree=3, method="lin"
):
    if not len(pa_i):
        return fit_marginal_mixture(
            MixingType.BASE_GMM, X, node_i, pa_i, range_k, resid, true_idl
        )
    if lg is not None and vb > 0:
        lg.info(f"Fitting mixture ({method})")

    import numpy as np
    import rpy2.robjects as robjects
    from rpy2.robjects import Formula, default_converter
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri

    with localconverter(default_converter + numpy2ri.converter):
        flexmix = importr("flexmix")
        splines = importr("splines")

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

        # build RHS terms depending on method
        rhs_terms = []
        for i in range(X_pa.shape[1]):
            xi = f"x.{i + 2}"
            if method == "quad":
                rhs_terms.append(f"poly({xi}, {2})")
            elif method == "cub":
                degree = 3
                rhs_terms.append(f"poly({xi}, {degree})")
            elif method == "ns":
                rhs_terms.append(f"ns({xi}, df={degree})")
            elif method == "bs":
                rhs_terms.append(f"bs({xi}, df={degree})")
            else:
                rhs_terms.append(xi)  # linear

        formula_str = "x.1 ~ " + " + ".join(rhs_terms)
        formula = Formula(formula_str)

        if lg is not None and vb > 0:
            lg.info(f"Formula: {formula}")

        best_bic = np.inf
        best_model = None
        best_k = None

        for k in range_k:
            m = flexmix.flexmix(formula, data=r_df, k=k)
            bic = robjects.r["BIC"](m)[0]
            if vb:
                print(f"k={k}, BIC={bic}")
            if bic < best_bic:
                best_bic = bic
                best_model = m
                best_k = k

        post_probs = np.array(robjects.r["posterior"](best_model))
        hard_assign = post_probs.argmax(axis=1)

    def post_entropy(p_proba, eps=1e-12):
        p_safe = np.clip(p_proba, eps, 1.0)
        return -np.sum(p_safe * np.log(p_safe), axis=1)

    ent_idl = post_entropy(post_probs.reshape(1, -1))
    res_dict = dict(
        bic=best_bic,
        idl=hard_assign,
        pproba=post_probs,
        entropy=ent_idl,
        best_k=best_k,
    )
    return res_dict


def conditional_mixture_known_assgn(X, node_i, pa_i, true_idl, **scoring_params):
    """ fit regresssions for a known mix assignment, pproba from log liks of those regressions (todo or degen?) """
    if len(pa_i) > 0:
        (Xx, y) = (X[:, pa_i], X[:, node_i])
        beta_l, sig_l = mix_regression_params_kn_assgn(Xx, y, true_idl)
        bic = mix_regression_bic(Xx, y, true_idl, beta_l, sig_l)
        pproba = None
        ent_idl = 0
    else:
        pproba = None
        bic = 0
        ent_idl = 0
    res_dict = dict(
        bic=bic,
        idl=true_idl,
        pproba=pproba,
        ent_idl=ent_idl
    )
    return res_dict
