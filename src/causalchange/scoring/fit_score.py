
from pygam import GAM


import numpy as np

from numpy.linalg import inv
from math import log

def fit_fun_IID(
        X, pa, target, score_fun, ret_residuals=False, **scoring_params):
    r""" fitting and scoring functional models

    :param X: parents
    :param y: target
    :param score_fun: callable, ret model, L
    :param scoring_params: hyperparameters

    :Keyword Arguments:
    * *score_type* (``ScoreType``) -- regressor and associated information-theoretic score
    """

    X_pa = np.random.normal(size=X[:, [target]].shape) if len(pa) == 0 else  X[:, pa]
    X_target = X[:, target]
    if ret_residuals:
        model, L, resid = score_fun(X_pa, X_target, return_residuals=True, **scoring_params)
        return resid, dict(model=model)
    else: model, L = score_fun(X_pa, X_target, return_residuals=ret_residuals, **scoring_params)
    return L, dict(model=model)


def fit_resid_CONTEXTS(X_dict, target, parents, score_fun, **score_params) -> list[np.ndarray]:
    """
    X_dict: {ctx_id: np.ndarray [n_ctx_samples, n_nodes]}
    """
    residual_sets = []
    for ctx_id, Xc in X_dict.items():
        y = Xc[:, target]
        if parents:
            X_pa = Xc[:, parents]
        else:
            # intercept-only model: one constant feature
            X_pa = np.ones((Xc.shape[0], 1), dtype=float)

        _, _, resid = score_fun(X_pa, y, return_residuals=True, **score_params)
        residual_sets.append(resid)

    return residual_sets

def fit_resid_TIME(X, target, parents, score_fun, changepoints, lag: int = 1, **score_params) -> list[np.ndarray]:
    """
    X: np.ndarray [T, n_nodes]
    changepoints: list of ints, including 0 and T, e.g., [0, 50, 100]
    lag: autoregressive lag to use for time-based parents
    """
    T, d = X.shape
    residual_sets = []

    for k in range(len(changepoints) - 1):
        start, end = changepoints[k], changepoints[k + 1]
        X_seg = X[start:end]      # [L, d]
        L = X_seg.shape[0]
        if L <= lag:
            continue  # not enough data for this segment

        # response at times lag..L-1
        y = X_seg[lag:, target]

        if parents:
            # here we take parents at the same time index (t); could be changed to lags if desired
            X_pa = X_seg[lag:, parents]
        else:
            # intercept-only: one constant feature
            X_pa = np.ones((L - lag, 1), dtype=float)

        _, _, res = score_fun(X_pa, y, return_residuals=True, **score_params)
        residual_sets.append(res)

    return residual_sets


def fit_resid_TIME_CONTEXTS(X_dict, target, parents, score_fun, cp_per_ctx, lag: int = 1, **score_params) -> list[np.ndarray]:
    """
    X_dict: {ctx_id: np.ndarray [T_ctx, n_nodes]}
    cp_per_ctx: {ctx_id: list of changepoints within that context's time axis}
    """
    residual_sets = []

    for ctx_id, Xc in X_dict.items():
        cps = list(cp_per_ctx[ctx_id])

        # ensure the changepoints cover the full range
        if len(cps) == 0 or cps[0] != 0:
            cps = [0] + cps
        if cps[-1] != Xc.shape[0]:
            cps = cps + [Xc.shape[0]]

        for k in range(len(cps) - 1):
            start, end = cps[k], cps[k + 1]
            X_seg = Xc[start:end]
            L = X_seg.shape[0]
            if L <= lag:
                continue

            y = X_seg[lag:, target]

            if parents:
                X_pa = X_seg[lag:, parents]
            else:
                # intercept-only
                X_pa = np.ones((L - lag, 1), dtype=float)

            _, _, res = score_fun(X_pa, y, return_residuals=True, **score_params)
            residual_sets.append(res)

    return residual_sets


def fit_score_gp(Xtr, ytr, return_residuals=False, **params):
    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float).reshape(-1)
    # guard: finite data
    if not np.all(np.isfinite(Xtr)) or not np.all(np.isfinite(ytr)):
        Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
        ytr = np.nan_to_num(ytr, nan=0.0, posinf=0.0, neginf=0.0)

    # standardize (improves conditioning); store scalers
    Xn, yn, scalers = _standardize(Xtr, ytr)
    n = Xn.shape[0]

    # hyperparam search config (unchanged)
    restarts = params.get("restarts", 10)
    low = params.get("bounds", {}).get("low", -5.0)
    high = params.get("bounds", {}).get("high", 5.0)
    refine = params.get("refine", True)
    rng = np.random.default_rng(params.get("seed", None))
    base_jitter = params.get("base_jitter", 1e-6)
    k_params = params.get("k_params", 3)
    use_bic = params.get("bic_penalty", False)

    # candidates (unchanged)
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

    # coarse search
    for th in cands:
        nll, cache, used_jit = eval_params(th)
        if nll < best_nll:
            best_nll, best, best_cache, best_jitter = nll, th, cache, used_jit

    # refine
    if refine and best is not None and np.isfinite(best_nll):
        for th in _grid_around(best, width=0.75, steps=3):
            nll, cache, used_jit = eval_params(th)
            if nll < best_nll:
                best_nll, best, best_cache, best_jitter = nll, th, cache, used_jit

    # fallback
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

    # unpack
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
        K_try.flat[:: n + 1] += (sn2 + jitter)  # add to diagonal
        try:
            # probe Cholesky just to test conditioning
            _ = np.linalg.cholesky(K_try)
            return K_try, jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    # last attempt: return with largest jitter; let caller handle failure in nll
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
    # BIC/MDL penalty
    penalty = 0.5 * k_params * np.log(max(n, 2))
    return (nll_nats + penalty) / np.log(2.0)

def _null_gaussian_mdl_bits(y):
    """Fallback: y ~ N(mu, sigma^2) i.i.d., MLE parameters; MDL in bits with BIC."""
    n = len(y)
    if n == 0:
        return 1e9  # degenerate
    mu = np.mean(y)
    var = np.var(y) + 1e-12
    nll = 0.5 * n * (np.log(2 * np.pi * var) + 1.0)  # exact MLE -loglik in nats
    k = 2  # mu, var
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
    """
    Kernel ridge regression with RBF kernel.

    Params:
      alpha: regularization strength (default 1.0)
      gamma: RBF kernel parameter (default 'scale' -> sklearn default)
      param_penalty: 'none' | 'rissanen' | 'bic' (default 'rissanen')
      fit_intercept: currently assumed False (we standardize X, but not y)
    """
    alpha = float(params.get("alpha", 1.0))
    gamma = params.get("gamma", None)  # None -> sklearn default
    param_penalty = params.get("param_penalty", "rissanen")

    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float).ravel()
    n = Xtr.shape[0]

    # Standardize features (like in linear case)
    scaler = StandardScaler()
    Xtr_std = scaler.fit_transform(Xtr)

    krr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    krr.fit(Xtr_std, ytr)
    yhat = krr.predict(Xtr_std)

    # We don't have a precise df for KernelRidge here.
    # As a simple proxy, set k = n (very conservative) or use 'none' penalty.
    k = n
    nlml_bits, rss, sigma2 = _gaussian_nlml_bits(ytr, yhat)
    score_bits = nlml_bits + _penalty_bits(param_penalty, k, n)

    # Wrap predict to include scaling
    def predict(Xte, return_var=False):
        Xte = np.asarray(Xte, float)
        Xte_std = scaler.transform(Xte)
        m = krr.predict(Xte_std)
        if return_var:
            # No analytic variance; return zeros (or a constant) for now.
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
import numpy as np
from math import log2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LinearRegression, Ridge
from scipy.special import comb


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
    # interactions is a list with the "order" per term; mirror the R code shape
    cost = 0.0
    for M in interactions:
        cost += slope_bits.logN(M) + _combinator(M, k) + M * np.log2(F)
    return float(cost)
def fit_score_spln(Xtr, ytr, return_residuals: bool = False, **params):
    """
    Python spline-MDL scorer mirroring the R 'earth' + GLOBE bits.

    Params (all optional):
      n_knots: int (default 10)
      degree: int (default 3)
      knots: 'quantile'|'uniform' (default 'quantile')
      extrapolation: 'continue'|'constant'|'linear'|'error' (default 'continue')
      model_type: 'ridge'|'ols' (default 'ridge')
      alpha: float (ridge strength, default 1.0)
      include_bias: bool (default False)
      globe_F: int (default 9)
    """
    X = np.asarray(Xtr, float)
    y = np.asarray(ytr, float).reshape(-1)
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
    model.fit(X, y)

    yhat = model.predict(X)
    resid = y - yhat
    sse = float(np.dot(resid, resid))

    # === build pieces to mirror the R path ===
    slope = _SlopeBits()
    k = np.array([d])             # number of parents/features
    dim = d
    rows = n
    mindiff = _min_diff(y)

    # "coeffs": concatenate spline internal knots + regression coefficients
    st = model.named_steps["splinetransformer"]
    knots_arr = getattr(st, "knots_", None)
    knots_flat = knots_arr.ravel() if knots_arr is not None else np.array([], dtype=float)
    coef = model.named_steps[list(model.named_steps.keys())[-1]].coef_
    coeffs_concat = np.concatenate([knots_flat, np.atleast_1d(coef).ravel()])

    model_bits = slope.model_score(coeffs_concat)

    # "hinges": number of basis functions produced
    Phi = model[:-1].transform(X)
    hinge_count = np.array([Phi.shape[1]], dtype=int)

    # "interactions": additive splines ⇒ order=1 for each basis term
    interactions = [1] * int(Phi.shape[1])

    base_cost = slope.model_score(k) + float(k[0]) * log2(dim if dim > 0 else 1)
    base_cost += slope.model_score(hinge_count)
    base_cost += _aggregate_hinges(interactions, int(k[0]), slope, globe_F)

    cost_bits = slope.gaussian_score_emp_sse(sse, rows, mindiff) + model_bits + base_cost

    def predict(Xte, return_var=False):
        ypred = model.predict(np.asarray(Xte, float))
        if return_var:
            # no Bayesian variance here; return empirical noise var as flat band
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
