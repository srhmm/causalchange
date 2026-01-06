from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import networkx as nx


@dataclass(frozen=True)
class SynthSpec:
    n_nodes: int = 10
    expected_degree: float = 2.0
    noise_std: float = 0.5
    weight_scale: float = 1.0


def random_dag(n_nodes: int, expected_degree: float, seed: int) -> nx.DiGraph:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_nodes).tolist()

    # expected out-degree ~ p*(n_nodes-1)/2 in forward-only model; use simple heuristic
    p = min(1.0, expected_degree / max(1, (n_nodes - 1) / 2))

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < p:
                u = order[i]
                v = order[j]
                G.add_edge(u, v)

    return G


def _sample_linear_sem_from_graph(
    G: nx.DiGraph,
    n_samples: int,
    noise_std: float,
    weight_scale: float,
    seed: int,
) -> np.ndarray:
    """
    Simple linear-Gaussian SEM: X_j = sum_{p in Pa(j)} w_pj * X_p + eps_j
    """
    rng = np.random.default_rng(seed)
    nodes = list(nx.topological_sort(G))
    N = len(nodes)

    # map to 0..N-1 indexing if needed (assuming nodes are 0..N-1 already in your code)
    X = np.zeros((n_samples, N), dtype=float)

    # weights per node
    W = {}
    for j in range(N):
        parents = sorted(list(G.predecessors(j)))
        if len(parents) == 0:
            W[j] = (parents, np.array([], dtype=float))
        else:
            w = rng.normal(loc=0.0, scale=weight_scale, size=len(parents))
            W[j] = (parents, w)

    for t in range(n_samples):
        for j in nodes:
            parents, w = W[j]
            pa_sum = float(X[t, parents] @ w) if parents else 0.0
            X[t, j] = pa_sum + noise_std * rng.normal()

    return X


def sample_iid(spec: SynthSpec, n_samples: int, seed: int) -> tuple[np.ndarray, nx.DiGraph]:
    G = random_dag(spec.n_nodes, spec.expected_degree, seed=seed)
    X = _sample_linear_sem_from_graph(
        G, n_samples=n_samples, noise_std=spec.noise_std, weight_scale=spec.weight_scale, seed=seed + 1
    )
    return X, G


def sample_contexts(
    spec: SynthSpec,
    n_contexts: int,
    n_samples_per_context: int,
    intervention_frac: float,
    intervention_strength: float,
    seed: int,
) -> tuple[Dict[int, np.ndarray], nx.DiGraph]:
    """
    Same DAG across contexts; context changes via shifting noise mean on a subset of nodes.
    """
    rng = np.random.default_rng(seed)
    G = random_dag(spec.n_nodes, spec.expected_degree, seed=seed)

    n_targets = max(1, int(round(intervention_frac * spec.n_nodes)))
    targets = rng.choice(spec.n_nodes, size=n_targets, replace=False)

    X_ctxs: Dict[int, np.ndarray] = {}
    for c in range(n_contexts):
        X = _sample_linear_sem_from_graph(
            G, n_samples=n_samples_per_context, noise_std=spec.noise_std, weight_scale=spec.weight_scale, seed=seed + 10 + c
        )
        # shift selected nodes (distribution change)
        shift = (c - (n_contexts - 1) / 2.0) * intervention_strength
        X[:, targets] += shift
        X_ctxs[c] = X

    return X_ctxs, G


def sample_mixed(
    spec: SynthSpec,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, nx.DiGraph]:
    """
    Very simple 2-mechanism mixture: for each node and sample, flip sign of weights.
    Returns X only (matching your current MIXED test usage).
    """
    rng = np.random.default_rng(seed)
    G = random_dag(spec.n_nodes, spec.expected_degree, seed=seed)
    N = spec.n_nodes
    nodes = list(nx.topological_sort(G))

    # base weights per node
    baseW = {}
    for j in range(N):
        parents = sorted(list(G.predecessors(j)))
        if not parents:
            baseW[j] = (parents, np.array([], dtype=float))
        else:
            w = rng.normal(loc=0.0, scale=spec.weight_scale, size=len(parents))
            baseW[j] = (parents, w)

    mech_assign = rng.integers(0, 2, size=(n_samples, N))  # 0/1 per sample and node
    X = np.zeros((n_samples, N), dtype=float)

    for t in range(n_samples):
        for j in nodes:
            parents, w = baseW[j]
            if parents:
                w_use = w if mech_assign[t, j] == 0 else -w
                pa_sum = float(X[t, parents] @ w_use)
            else:
                pa_sum = 0.0
            X[t, j] = pa_sum + spec.noise_std * rng.normal()

    return X, G


def sample_time_series(
    spec: SynthSpec,
    n_timepoints: int,
    seed: int,
) -> tuple[np.ndarray, nx.DiGraph]:
    """
    VAR(1) process consistent with a DAG skeleton (still not truly "instantaneous causal DAG",
    but matches your (T, N) expectation for TIME mode).
    """
    rng = np.random.default_rng(seed)
    G = random_dag(spec.n_nodes, spec.expected_degree, seed=seed)
    N = spec.n_nodes

    A = np.zeros((N, N), dtype=float)
    for (u, v) in G.edges():
        A[u, v] = rng.normal(loc=0.0, scale=0.15)  # keep small for stability

    burn_in = 50
    T_total = n_timepoints + burn_in
    X = np.zeros((T_total, N), dtype=float)

    X[0, :] = spec.noise_std * rng.normal(size=N)
    for t in range(1, T_total):
        X[t, :] = X[t - 1, :] @ A + spec.noise_std * rng.normal(size=N)

    return X[burn_in:, :], G


def sample_time_series_contexts(
    spec: SynthSpec,
    n_contexts: int,
    n_timepoints_per_context: int,
    seed: int,
) -> tuple[Dict[int, np.ndarray], nx.DiGraph]:
    """
    Same VAR dynamics across contexts but with context-specific mean shifts on subset of nodes.
    """
    rng = np.random.default_rng(seed)
    X0, G = sample_time_series(spec, n_timepoints=n_timepoints_per_context, seed=seed)

    n_targets = max(1, int(round(0.2 * spec.n_nodes)))
    targets = rng.choice(spec.n_nodes, size=n_targets, replace=False)

    X_ctxs = {}
    for c in range(n_contexts):
        Xc, _ = sample_time_series(spec, n_timepoints=n_timepoints_per_context, seed=seed + 100 + c)
        Xc[:, targets] += (c - (n_contexts - 1) / 2.0) * 0.8
        X_ctxs[c] = Xc

    return X_ctxs, G
