import numpy as np
def sample_linear_sem(adj, n_samples=200, noise_std=0.1, seed=0):
    rng = np.random.default_rng(seed)
    d = adj.shape[0]
    X = np.zeros((n_samples, d))
    order = list(range(d))

    for t in range(n_samples):
        for j in order:
            parents = np.where(adj[:, j] != 0)[0]
            pa_sum = X[t, parents].sum() if len(parents) > 0 else 0.0
            X[t, j] = pa_sum + noise_std * rng.normal()
    return X
