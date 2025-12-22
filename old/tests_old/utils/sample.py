# tests/conftest.py
import numpy as np
import pytest


def sample_linear_sem(adj, n_samples=200, noise_std=0.1, seed=0):
    rng = np.random.default_rng(seed)
    N = adj.shape[0]
    D = n_samples
    X = np.zeros((D, N))
    order = list(range(N))  # assume topological order

    for t in range(D):
        for j in order:
            parents = np.where(adj[:, j] != 0)[0]
            pa_sum = X[t, parents].sum() if len(parents) > 0 else 0.0
            X[t, j] = pa_sum + noise_std * rng.normal()
    return X
import numpy as np

def sample_linear_sem_mixed(adj, n_samples=300, noise_std=0.2, seed=0):
    rng = np.random.default_rng(seed)
    N = adj.shape[0]
    D = n_samples

    weights0 = {}
    weights1 = {}
    for j in range(N):
        parents = np.where(adj[:, j] != 0)[0]
        w0 = rng.normal(1.0, 0.2, size=len(parents))   # mechanism A
        w1 = rng.normal(-1.0, 0.2, size=len(parents))  # mechanism B
        weights0[j] = (parents, w0)
        weights1[j] = (parents, w1)

    mech_assign = rng.integers(0, 2, size=(D, N))  # entries ∈ {0,1}

    X = np.zeros((D, N))
    for t in range(D):
        for j in range(N):
            parents = weights0[j][0]
            if mech_assign[t, j] == 0:
                w = weights0[j][1]
            else:
                w = weights1[j][1]
            pa_sum = X[t, parents] @ w if len(parents) > 0 else 0
            X[t, j] = pa_sum + noise_std * rng.normal()

    return X, mech_assign



def sample_time_series(adj_or_lagged,
                       n_timepoints=300,
                       noise_std=0.1,
                       seed=0):
    rng = np.random.default_rng(seed)

    if isinstance(adj_or_lagged, dict):
        adj_lagged = adj_or_lagged
    else:
        A = np.asarray(adj_or_lagged)
        adj_lagged = {1: A}

    lags = sorted(adj_lagged.keys())
    max_lag = max(lags)
    N = next(iter(adj_lagged.values())).shape[0]

    for lag, A_lag in adj_lagged.items():
        A_lag = np.asarray(A_lag)
        assert A_lag.shape == (N, N), f"Adjacency at lag {lag} has wrong shape {A_lag.shape}"

    burn_in = max_lag * 5
    T_total = n_timepoints + burn_in

    X = np.zeros((T_total, N))

    X[:max_lag, :] = noise_std * rng.normal(size=(max_lag, N))

    for t in range(max_lag, T_total):
        for lag in lags:
            A_lag = adj_lagged[lag]
            X[t, :] += X[t - lag, :] @ A_lag
        X[t, :] += noise_std * rng.normal(size=N)

    X = X[burn_in:, :]
    assert X.shape == (n_timepoints, N)
    return X

def sample_time_series_contexts(adj_or_lagged,
                                n_contexts=2,
                                n_timepoints_per_context=200,
                                noise_std=0.1,
                                seed=0):
    X_contexts = {}
    for ctx in range(n_contexts):
        X_ctx = sample_time_series(
            adj_or_lagged=adj_or_lagged,
            n_timepoints=n_timepoints_per_context,
            noise_std=noise_std,
            seed=seed + ctx,
        )
        X_contexts[ctx] = X_ctx
    return X_contexts

@pytest.fixture
def rng():
    return np.random.default_rng(42)
