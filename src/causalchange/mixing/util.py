import numpy as np
from sklearn.linear_model import LinearRegression


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

        if len(y_k) == 0: continue  # skip empty

        residuals = y_k - X_k @ beta_k
        log_likelihood += np.sum(-0.5 * np.log(2 * np.pi * sigma_k ** 2) - 0.5 * (residuals ** 2) / sigma_k ** 2)
        log_likelihood += len(y_k) * np.log(
            mixture_weights[k] + 1e-12)  # log weight f each pt in cluster, small eps for stability

    # n parameters:
    # -beta_k: D per component
    # -sigma_k: 1 per component
    #  -mixture weights: K-1 (last one is determined)
    num_params = K * D + K + (K - 1)

    bic = -2 * log_likelihood + num_params * np.log(N)
    return bic
