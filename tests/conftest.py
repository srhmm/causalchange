# tests/conftest.py
import numpy as np
import pytest


def sample_linear_sem(adj, n_samples=200, noise_std=0.1, seed=0):
    """
    Simple linear SEM generator.

    adj: (N, N) adjacency matrix of a DAG over N nodes
    Returns:
        X: (D, N) = (n_samples, n_nodes)

    Assumes nodes are in a valid topological order (0..N-1).
    """
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
    """
    Generate data for DataMode.MIXED:
    - X is (D, N)
    - Each node has TWO mechanisms (two parent weight vectors)
    - Each sample independently picks mechanism 0 or 1 for each node
    """
    rng = np.random.default_rng(seed)
    N = adj.shape[0]
    D = n_samples

    # Two mechanisms per node: random weights depending on parents
    weights0 = {}
    weights1 = {}
    for j in range(N):
        parents = np.where(adj[:, j] != 0)[0]
        w0 = rng.normal(1.0, 0.2, size=len(parents))   # mechanism A
        w1 = rng.normal(-1.0, 0.2, size=len(parents))  # mechanism B
        weights0[j] = (parents, w0)
        weights1[j] = (parents, w1)

    # For each sample and node choose mechanism 0 or 1
    mech_assign = rng.integers(0, 2, size=(D, N))  # entries ∈ {0,1}

    # Generate data
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

    return X, mech_assign  # return assignment too (useful later)


import numpy as np


def sample_time_series(adj_or_lagged,
                       n_timepoints=300,
                       noise_std=0.1,
                       seed=0):
    """
    Sample a multivariate time series from a linear structural causal model (SCM)
    with time-lagged dependencies, similar in spirit to tigramite.

    Parameters
    ----------
    adj_or_lagged : np.ndarray or dict[int, np.ndarray]
        - If np.ndarray of shape (N, N):
            Interpreted as lag-1 adjacency matrix B^{(1)}.
            Edge i->j means X_t^j depends linearly on X_{t-1}^i.
        - If dict: {lag: A_lag}, where each A_lag is (N, N):
            Edge i->j in A_lag means X_t^j depends on X_{t-lag}^i.

        All adjacency matrices are interpreted as *coefficient matrices* directly.

    n_timepoints : int
        Number of time points T (this will be your D).

    noise_std : float
        Standard deviation of the Gaussian noise ε_t^j.

    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (T, N)
        Simulated time series, where T = n_timepoints, N = num variables.
        This can be passed directly to CausalChange with DataMode.TIME
        (D = T, N = num nodes).
    """
    rng = np.random.default_rng(seed)

    # Normalize inputs: allow either single adjacency or dict of lagged ones
    if isinstance(adj_or_lagged, dict):
        adj_lagged = adj_or_lagged
    else:
        A = np.asarray(adj_or_lagged)
        adj_lagged = {1: A}

    # Basic shapes
    lags = sorted(adj_lagged.keys())
    max_lag = max(lags)
    N = next(iter(adj_lagged.values())).shape[0]

    # Sanity check: all lag matrices have same shape
    for lag, A_lag in adj_lagged.items():
        A_lag = np.asarray(A_lag)
        assert A_lag.shape == (N, N), f"Adjacency at lag {lag} has wrong shape {A_lag.shape}"

    # We'll simulate with some burn-in and discard initial transients
    burn_in = max_lag * 5
    T_total = n_timepoints + burn_in

    X = np.zeros((T_total, N))

    # Initialize first max_lag time points with pure noise
    X[:max_lag, :] = noise_std * rng.normal(size=(max_lag, N))

    # SCM recursion: for t >= max_lag,
    # X_t^j = sum_{lag} sum_i A_lag[i,j] * X_{t-lag}^i + eps_t^j
    for t in range(max_lag, T_total):
        for lag in lags:
            A_lag = adj_lagged[lag]
            X[t, :] += X[t - lag, :] @ A_lag  # (N,) @ (N, N) -> (N,)
        X[t, :] += noise_std * rng.normal(size=N)

    # Discard burn-in
    X = X[burn_in:, :]
    assert X.shape == (n_timepoints, N)
    return X

def sample_time_series_contexts(adj_or_lagged,
                                n_contexts=2,
                                n_timepoints_per_context=200,
                                noise_std=0.1,
                                seed=0):
    """
    Sample multiple-context time series data for DataMode.TIME_CONTEXTS.

    Parameters
    ----------
    adj_or_lagged : np.ndarray or dict[int, np.ndarray]
        Same as in sample_time_series(..).

    n_contexts : int
        Number of separate time series (contexts).

    n_timepoints_per_context : int
        Number of time points per context.

    noise_std : float
        Noise std, passed to sample_time_series.

    seed : int
        Global RNG seed; each context gets seed + ctx_id.

    Returns
    -------
    X_contexts : dict[int, np.ndarray]
        Mapping context_id -> X_ctx, where X_ctx is (T_c, N).
        This can be passed directly to CausalChange with DataMode.TIME_CONTEXTS.
    """
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
    """Optional fixture if you want a shared RNG in tests."""
    return np.random.default_rng(42)
