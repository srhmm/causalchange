from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from causalchange.config.benchmark_config import SingleDataConfig, MultiDataConfig


def sample_single_continuous(config: SingleDataConfig):
    sampling_fun = sample_linear_gaussian if config.nonlinearity == "lin" else sample_nonlinear_additive
    return sampling_fun(
        n_samples=config.n_samples,
        n_nodes=config.n_nodes,
        edge_prob=config.edge_prob,
        seed=config.seed,
        weight_scale=config.weight_scale,
        noise_scale=config.noise_scale,
    )

def sample_multi_continuous(config: MultiDataConfig):
    sampling_fun = sample_multicontext_linear_gaussian_interventional if config.nonlinearity == "lin" else sample_multicontext_nonlinear_additive_interventional

    return sampling_fun(
        n_samples_per_context=config.n_samples_per_context,
        n_nodes=config.n_nodes,
        edge_prob=config.edge_prob,
        n_contexts=config.n_contexts,
        seed=config.seed,
        context_col=config.context_col,
        n_intervened_per_context=config.n_intervened_per_context,
        weight_scale=config.weight_scale,
        noise_scale=config.noise_scale,
        intervention_type=config.intervention_type,
        weight_scale_intervened=config.weight_scale_intervened,
        shift_scale=config.shift_scale,
        noise_scale_intervened=config.noise_scale_intervened,
    )

def sample_single_temporal(config: SingleDataConfig):
    sampling_fun = sample_temporal_linear if config.nonlinearity == "lin" else sample_temporal_nonlinear


    return sampling_fun(
        n_samples=config.n_samples,
        n_nodes=config.n_nodes,
        edge_prob=config.edge_prob,
        seed=config.seed,
        weight_scale=config.weight_scale,
        noise_scale=config.noise_scale,
    )
def sample_multi_temporal(config: SingleDataConfig):
    sampling_fun = sample_multicontext_temporal_linear if config.nonlinearity == "lin" else sample_multicontext_temporal_nonlinear
    return sampling_fun(
        n_samples_per_context=config.n_samples_per_context,
        n_nodes=config.n_nodes,
        edge_prob=config.edge_prob,
        n_contexts=config.n_contexts,
        seed=config.seed,
        context_col=config.context_col,
        intervention_type=config.intervention_type,
        n_intervened_per_context=config.n_intervened_per_context,
    )


def _random_dag(n_nodes: int, edge_prob: float, rng: np.random.Generator) -> nx.DiGraph:
    order = rng.permutation(n_nodes).tolist()
    pos = {node: i for i, node in enumerate(order)}
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            if pos[i] < pos[j] and rng.random() < edge_prob:
                g.add_edge(i, j)
    return g

def sample_nonlinear_additive(
    *,
    n_samples: int,
    n_nodes: int,
    edge_prob: float,
    seed: int,
    weight_scale: float = 2.0,
    noise_scale: float = 0.7,
    nonlinearity: str = "tanh",  # "tanh" | "sin" | "relu"
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    rng = np.random.default_rng(seed)
    g = _random_dag(n_nodes=n_nodes, edge_prob=edge_prob, rng=rng)

    W = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v in g.edges():
        w = rng.normal(loc=0.0, scale=weight_scale)
        if abs(w) < 0.1:
            w = 0.1 if w >= 0 else -0.1
        W[u, v] = w

    if nonlinearity == "tanh":
        f = np.tanh
    elif nonlinearity == "sin":
        f = np.sin
    elif nonlinearity == "relu":
        f = lambda x: np.maximum(x, 0.0)
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

    topo = list(nx.topological_sort(g))
    X = np.zeros((n_samples, n_nodes), dtype=float)
    eps = rng.normal(loc=0.0, scale=noise_scale, size=(n_samples, n_nodes))

    for v in topo:
        parents = list(g.predecessors(v))
        if parents:
            lin = X[:, parents] @ W[parents, v]
            X[:, v] = f(lin) + eps[:, v]
        else:
            X[:, v] = eps[:, v]

    cols = [f"X{i}" for i in range(n_nodes)]
    df = pd.DataFrame(X, columns=cols)
    mapping = {i: cols[i] for i in range(n_nodes)}
    g_named = nx.relabel_nodes(g, mapping, copy=True)
    return df, g_named


def sample_linear_gaussian(
    *,
    n_samples: int,
    n_nodes: int,
    edge_prob: float,
    seed: int,
    weight_scale: float = 2.0,
    noise_scale: float = 0.7,
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    rng = np.random.default_rng(seed)
    g = _random_dag(n_nodes=n_nodes, edge_prob=edge_prob, rng=rng)

    W = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v in g.edges():
        w = rng.normal(loc=0.0, scale=weight_scale)
        if abs(w) < 0.1:
            w = 0.1 if w >= 0 else -0.1
        W[u, v] = w

    topo = list(nx.topological_sort(g))
    X = np.zeros((n_samples, n_nodes), dtype=float)
    eps = rng.normal(loc=0.0, scale=noise_scale, size=(n_samples, n_nodes))

    for v in topo:
        parents = list(g.predecessors(v))
        if parents:
            X[:, v] = X[:, parents] @ W[parents, v] + eps[:, v]
        else:
            X[:, v] = eps[:, v]

    cols = [f"X{i}" for i in range(n_nodes)]
    df = pd.DataFrame(X, columns=cols)
    mapping = {i: cols[i] for i in range(n_nodes)}
    g_named = nx.relabel_nodes(g, mapping, copy=True)
    return df, g_named



def _cut_incoming_edges(g: nx.DiGraph, targets: List[int]) -> nx.DiGraph:
    gc = g.copy()
    for t in targets:
        for p in list(gc.predecessors(t)):
            gc.remove_edge(p, t)
    return gc


def _intersect_edges(graphs: List[nx.DiGraph]) -> nx.DiGraph:
    if not graphs:
        raise ValueError("graphs must be non-empty")
    nodes = list(graphs[0].nodes())
    edge_sets = [set(gr.edges()) for gr in graphs]
    common = set.intersection(*edge_sets) if edge_sets else set()
    out = nx.DiGraph()
    out.add_nodes_from(nodes)
    out.add_edges_from(list(common))
    return out


def _sample_from_weighted_dag_linear(
    g: nx.DiGraph,
    *,
    n_samples: int,
    W: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    n_nodes = len(g.nodes())
    topo = list(nx.topological_sort(g))
    X = np.zeros((n_samples, n_nodes), dtype=float)
    eps = rng.normal(loc=0.0, scale=noise_scale, size=(n_samples, n_nodes))

    for v in topo:
        parents = list(g.predecessors(v))
        if parents:
            X[:, v] = X[:, parents] @ W[parents, v] + eps[:, v]
        else:
            X[:, v] = eps[:, v]
    return X


def _sample_from_weighted_dag_nonlinear(
    g: nx.DiGraph,
    *,
    n_samples: int,
    W: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float,
    nonlinearity: str,
) -> np.ndarray:
    if nonlinearity == "tanh":
        f = np.tanh
    elif nonlinearity == "sin":
        f = np.sin
    elif nonlinearity == "relu":
        f = lambda x: np.maximum(x, 0.0)
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

    n_nodes = len(g.nodes())
    topo = list(nx.topological_sort(g))
    X = np.zeros((n_samples, n_nodes), dtype=float)
    eps = rng.normal(loc=0.0, scale=noise_scale, size=(n_samples, n_nodes))

    for v in topo:
        parents = list(g.predecessors(v))
        if parents:
            lin = X[:, parents] @ W[parents, v]
            X[:, v] = f(lin) + eps[:, v]
        else:
            X[:, v] = eps[:, v]
    return X

def sample_multicontext_linear_gaussian_interventional(
    *,
    n_samples_per_context: int,
    n_nodes: int,
    edge_prob: float,
    n_contexts: int,
    seed: int,
    context_col: str = "context",
    n_intervened_per_context: int = 1,
    weight_scale: float = 2.0,
    noise_scale: float = 0.7,
    intervention_type: str = "soft_weight",  # "hard"|"soft_weight"|"shift"|"noise"
    weight_scale_intervened: float = 2.0,
    shift_scale: float = 2.0,
    noise_scale_intervened: float | None = None,
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    rng = np.random.default_rng(seed)
    g_base = _random_dag(n_nodes=n_nodes, edge_prob=edge_prob, rng=rng)

    W_base = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v in g_base.edges():
        w = rng.normal(loc=0.0, scale=weight_scale)
        if abs(w) < 0.1:
            w = 0.1 if w >= 0 else -0.1
        W_base[u, v] = w

    if intervention_type not in {"hard", "soft_weight", "shift", "noise"}:
        raise ValueError(f"Unknown intervention_type: {intervention_type}")

    contexts = list(range(n_contexts))
    context_graphs: Dict[int, nx.DiGraph] = {}
    interventions: Dict[int, List[str]] = {}
    blocks = []

    cols = [f"X{i}" for i in range(n_nodes)]
    mapping = {i: cols[i] for i in range(n_nodes)}

    for c in contexts:
        targets = rng.choice(n_nodes, size=min(n_intervened_per_context, n_nodes), replace=False).tolist()

        if intervention_type == "hard":
            g_c = _cut_incoming_edges(g_base, targets)
        else:
            g_c = g_base

        # context-specific weights / shifts / noise
        W_c = W_base.copy()
        shift = np.zeros(n_nodes, dtype=float)
        noise_vec = np.full(n_nodes, noise_scale, dtype=float)

        if intervention_type == "soft_weight":
            for t in targets:
                for p in list(g_base.predecessors(t)):
                    W_c[p, t] = W_c[p, t] * weight_scale_intervened

        elif intervention_type == "shift":
            for t in targets:
                shift[t] = rng.normal(loc=0.0, scale=shift_scale)

        elif intervention_type == "noise":
            ns = noise_scale_intervened if noise_scale_intervened is not None else (2.0 * noise_scale)
            for t in targets:
                noise_vec[t] = ns

        topo = list(nx.topological_sort(g_c))
        X = np.zeros((n_samples_per_context, n_nodes), dtype=float)

        # per-node noise
        eps = rng.normal(loc=0.0, scale=1.0, size=(n_samples_per_context, n_nodes))
        eps = eps * noise_vec[None, :]

        for v in topo:
            parents = list(g_c.predecessors(v))
            if parents:
                X[:, v] = X[:, parents] @ W_c[parents, v] + shift[v] + eps[:, v]
            else:
                X[:, v] = shift[v] + eps[:, v]

        df_c = pd.DataFrame(X, columns=cols)
        df_c[context_col] = c
        blocks.append(df_c)

        context_graphs[c] = nx.relabel_nodes(g_c, mapping, copy=True)
        interventions[c] = [mapping[t] for t in targets]

    df = pd.concat(blocks, ignore_index=True)

    true_base = nx.relabel_nodes(g_base, mapping, copy=True)
    true_target = _true_target_graph(true_base, context_graphs, intervention_type)
    return df, true_target #df, true_base, true_target, context_graphs, interventions

def sample_multicontext_nonlinear_additive_interventional(
    *,
    n_samples_per_context: int,
    n_nodes: int,
    edge_prob: float,
    n_contexts: int,
    seed: int,
    context_col: str = "context",
    n_intervened_per_context: int = 1,
    weight_scale: float = 2.0,
    noise_scale: float = 0.7,
    nonlinearity: str = "tanh",
    intervention_type: str = "soft_mechanism",  # "hard"|"soft_weight"|"soft_mechanism"|"shift"|"noise"
    weight_scale_intervened: float = 2.0,
    alt_nonlinearity: str = "sin",
    shift_scale: float = 2.0,
    noise_scale_intervened: float | None = None,
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    rng = np.random.default_rng(seed)
    g_base = _random_dag(n_nodes=n_nodes, edge_prob=edge_prob, rng=rng)

    W_base = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v in g_base.edges():
        w = rng.normal(loc=0.0, scale=weight_scale)
        if abs(w) < 0.1:
            w = 0.1 if w >= 0 else -0.1
        W_base[u, v] = w

    if intervention_type not in {"hard", "soft_weight", "soft_mechanism", "shift", "noise"}:
        raise ValueError(f"Unknown intervention_type: {intervention_type}")

    # activation functions
    def _act(name: str):
        if name == "tanh":
            return np.tanh
        if name == "sin":
            return np.sin
        if name == "relu":
            return lambda x: np.maximum(x, 0.0)
        raise ValueError(f"Unknown nonlinearity: {name}")

    f_base = _act(nonlinearity)
    f_alt = _act(alt_nonlinearity)

    contexts = list(range(n_contexts))
    context_graphs: Dict[int, nx.DiGraph] = {}
    interventions: Dict[int, List[str]] = {}
    blocks = []

    cols = [f"X{i}" for i in range(n_nodes)]
    mapping = {i: cols[i] for i in range(n_nodes)}

    for c in contexts:
        targets = rng.choice(n_nodes, size=min(n_intervened_per_context, n_nodes), replace=False).tolist()

        if intervention_type == "hard":
            g_c = _cut_incoming_edges(g_base, targets)
        else:
            g_c = g_base

        W_c = W_base.copy()
        shift = np.zeros(n_nodes, dtype=float)
        noise_vec = np.full(n_nodes, noise_scale, dtype=float)

        # node-specific mechanisms: default f_base, optionally f_alt for targets
        node_f = [f_base for _ in range(n_nodes)]

        if intervention_type == "soft_weight":
            for t in targets:
                for p in list(g_base.predecessors(t)):
                    W_c[p, t] = W_c[p, t] * weight_scale_intervened

        elif intervention_type == "soft_mechanism":
            for t in targets:
                node_f[t] = f_alt

        elif intervention_type == "shift":
            for t in targets:
                shift[t] = rng.normal(loc=0.0, scale=shift_scale)

        elif intervention_type == "noise":
            ns = noise_scale_intervened if noise_scale_intervened is not None else (2.0 * noise_scale)
            for t in targets:
                noise_vec[t] = ns

        topo = list(nx.topological_sort(g_c))
        X = np.zeros((n_samples_per_context, n_nodes), dtype=float)

        eps = rng.normal(loc=0.0, scale=1.0, size=(n_samples_per_context, n_nodes))
        eps = eps * noise_vec[None, :]

        for v in topo:
            parents = list(g_c.predecessors(v))
            if parents:
                lin = X[:, parents] @ W_c[parents, v]
                X[:, v] = node_f[v](lin) + shift[v] + eps[:, v]
            else:
                X[:, v] = shift[v] + eps[:, v]

        df_c = pd.DataFrame(X, columns=cols)
        df_c[context_col] = c
        blocks.append(df_c)

        context_graphs[c] = nx.relabel_nodes(g_c, mapping, copy=True)
        interventions[c] = [mapping[t] for t in targets]

    df = pd.concat(blocks, ignore_index=True)

    true_base = nx.relabel_nodes(g_base, mapping, copy=True)
    true_target = _true_target_graph(true_base, context_graphs, intervention_type)
    return df, true_target #df, true_base, true_target, context_graphs, interventions

def _true_target_graph(
    true_base: nx.DiGraph,
    context_graphs: Dict[int, nx.DiGraph],
    intervention_type: str,
) -> nx.DiGraph:
        return true_base
    #if intervention_type == "hard":
    #    return _intersect_edges(list(context_graphs.values()))
    #return true_base


def _act(name: str):
    if name == "tanh":
        return np.tanh
    if name == "sin":
        return np.sin
    if name == "relu":
        return lambda x: np.maximum(x, 0.0)
    if name == "lin":
        return lambda x: x
    raise ValueError(f"Unknown nonlinearity: {name}")


def _make_lag_mats(
    *,
    n_nodes: int,
    tau_max: int,
    edge_prob: float,
    rng: np.random.Generator,
    weight_scale: float,
) -> List[np.ndarray]:
    """Create lagged coefficient matrices A_l for l=1..tau_max.
    A_l[u, v] is weight from X_u(t-l) to X_v(t).
    """
    mats: List[np.ndarray] = []
    for _lag in range(1, tau_max + 1):
        A = np.zeros((n_nodes, n_nodes), dtype=float)
        mask = rng.random((n_nodes, n_nodes)) < edge_prob
        np.fill_diagonal(mask, False)
        idx = np.argwhere(mask)
        for u, v in idx:
            w = rng.normal(loc=0.0, scale=weight_scale)
            if abs(w) < 0.1:
                w = 0.1 if w >= 0 else -0.1
            A[u, v] = w
        mats.append(A)
    return mats


def _temporal_true_graph(
    *,
    cols: List[str],
    tau_max: int,
    g_inst: nx.DiGraph,
    A_lags: List[np.ndarray],
) -> nx.DiGraph:
    """Temporal graph with node tuples (var, lag)."""
    out = nx.DiGraph()
    for v in cols:
        for lag in range(0, tau_max + 1):
            out.add_node((v, lag))

    # instantaneous edges: (Xj,0)->(Xi,0)
    for u, v in g_inst.edges():
        out.add_edge((cols[u], 0), (cols[v], 0))

    # lagged edges: (Xj,lag)->(Xi,0)
    for lag, A in enumerate(A_lags, start=1):
        uu, vv = np.nonzero(A)
        for u, v in zip(uu.tolist(), vv.tolist()):
            out.add_edge((cols[u], lag), (cols[v], 0))

    return out


def _simulate_temporal_sem(
    *,
    n_samples: int,
    cols: List[str],
    tau_max: int,
    topo_inst: List[int],
    inst_parents: List[List[int]],
    W_inst: np.ndarray,
    A_lags: List[np.ndarray],
    rng: np.random.Generator,
    noise_scale: np.ndarray,         # shape (n_nodes,)
    shift: np.ndarray,               # shape (n_nodes,)
    node_act: List[callable],        # len n_nodes
    burnin: int,
) -> np.ndarray:
    n_nodes = len(cols)
    T = n_samples + burnin + tau_max

    X = np.zeros((T, n_nodes), dtype=float)

    # initialize history (helps stability)
    X[:tau_max, :] = rng.normal(loc=0.0, scale=1.0, size=(tau_max, n_nodes))

    eps = rng.normal(loc=0.0, scale=1.0, size=(T, n_nodes)) * noise_scale[None, :]

    for t in range(tau_max, T):
        # lagged contribution for each node at time t
        lag_contrib = np.zeros(n_nodes, dtype=float)
        for lag in range(1, tau_max + 1):
            A = A_lags[lag - 1]
            lag_contrib += X[t - lag, :] @ A  # shape (n_nodes,)

        # instantaneous SEM in topological order
        x_t = np.zeros(n_nodes, dtype=float)
        for v in topo_inst:
            pa = inst_parents[v]
            inst = 0.0
            if pa:
                inst = float(x_t[pa] @ W_inst[pa, v])
            lin = lag_contrib[v] + inst
            x_t[v] = float(node_act[v](lin) + shift[v] + eps[t, v])

        X[t, :] = x_t

    # drop burnin + initial history
    X_out = X[(tau_max + burnin) : (tau_max + burnin + n_samples), :]
    return X_out


def sample_temporal_linear(
    *,
    n_samples: int,
    n_nodes: int,
    edge_prob: float,
    seed: int,
    tau_max: int = 1,
    weight_scale: float = 0.3,
    noise_scale: float = 0.7,
    burnin: int | None = None,
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    rng = np.random.default_rng(seed)
    burnin = int(10 * tau_max) if burnin is None else int(burnin)

    cols = [f"X{i}" for i in range(n_nodes)]

    # instantaneous DAG + weights
    g_inst = _random_dag(n_nodes=n_nodes, edge_prob=edge_prob, rng=rng)  # reuse your helper :contentReference[oaicite:2]{index=2}
    W_inst = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v in g_inst.edges():
        w = rng.normal(loc=0.0, scale=weight_scale)
        if abs(w) < 0.1:
            w = 0.1 if w >= 0 else -0.1
        W_inst[u, v] = w

    topo_inst = list(nx.topological_sort(g_inst))
    inst_parents = [list(g_inst.predecessors(v)) for v in range(n_nodes)]

    # lagged matrices
    A_lags = _make_lag_mats(
        n_nodes=n_nodes, tau_max=tau_max, edge_prob=edge_prob,
        rng=rng, weight_scale=weight_scale,
    )

    node_act = [_act("lin") for _ in range(n_nodes)]
    noise_vec = np.full(n_nodes, float(noise_scale), dtype=float)
    shift = np.zeros(n_nodes, dtype=float)

    X = _simulate_temporal_sem(
        n_samples=n_samples,
        cols=cols,
        tau_max=tau_max,
        topo_inst=topo_inst,
        inst_parents=inst_parents,
        W_inst=W_inst,
        A_lags=A_lags,
        rng=rng,
        noise_scale=noise_vec,
        shift=shift,
        node_act=node_act,
        burnin=burnin,
    )

    df = pd.DataFrame(X, columns=cols)
    true_g = _temporal_true_graph(cols=cols, tau_max=tau_max, g_inst=g_inst, A_lags=A_lags)
    return df, true_g


def sample_temporal_nonlinear(
    *,
    n_samples: int,
    n_nodes: int,
    edge_prob: float,
    seed: int,
    tau_max: int = 1,
    weight_scale: float = 0.3,
    noise_scale: float = 0.7,
    nonlinearity: str = "tanh",
    burnin: int | None = None,
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    rng = np.random.default_rng(seed)
    burnin = int(10 * tau_max) if burnin is None else int(burnin)

    cols = [f"X{i}" for i in range(n_nodes)]

    g_inst = _random_dag(n_nodes=n_nodes, edge_prob=edge_prob, rng=rng)  # :contentReference[oaicite:3]{index=3}
    W_inst = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v in g_inst.edges():
        w = rng.normal(loc=0.0, scale=weight_scale)
        if abs(w) < 0.1:
            w = 0.1 if w >= 0 else -0.1
        W_inst[u, v] = w

    topo_inst = list(nx.topological_sort(g_inst))
    inst_parents = [list(g_inst.predecessors(v)) for v in range(n_nodes)]

    A_lags = _make_lag_mats(
        n_nodes=n_nodes, tau_max=tau_max, edge_prob=edge_prob,
        rng=rng, weight_scale=weight_scale,
    )

    f = _act(nonlinearity)
    node_act = [f for _ in range(n_nodes)]
    noise_vec = np.full(n_nodes, float(noise_scale), dtype=float)
    shift = np.zeros(n_nodes, dtype=float)

    X = _simulate_temporal_sem(
        n_samples=n_samples,
        cols=cols,
        tau_max=tau_max,
        topo_inst=topo_inst,
        inst_parents=inst_parents,
        W_inst=W_inst,
        A_lags=A_lags,
        rng=rng,
        noise_scale=noise_vec,
        shift=shift,
        node_act=node_act,
        burnin=burnin,
    )

    df = pd.DataFrame(X, columns=cols)
    true_g = _temporal_true_graph(cols=cols, tau_max=tau_max, g_inst=g_inst, A_lags=A_lags)
    return df, true_g


def sample_multicontext_temporal_linear(
    *,
    n_samples_per_context: int,
    n_nodes: int,
    edge_prob: float,
    n_contexts: int,
    seed: int,
    tau_max: int = 1,
    context_col: str = "context",
    n_intervened_per_context: int = 1,
    intervention_type: str = "soft_weight",  # "hard"|"soft_weight"|"shift"|"noise"
    weight_scale: float = 0.3,
    noise_scale: float = 0.7,
    weight_scale_intervened: float = 2.0,
    shift_scale: float = 2.0,
    noise_scale_intervened: float | None = None,
    burnin: int | None = None,
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    rng = np.random.default_rng(seed)
    burnin = int(10 * tau_max) if burnin is None else int(burnin)

    cols = [f"X{i}" for i in range(n_nodes)]
    mapping = {i: cols[i] for i in range(n_nodes)}

    # base instantaneous DAG + weights
    g_inst = _random_dag(n_nodes=n_nodes, edge_prob=edge_prob, rng=rng)  # :contentReference[oaicite:4]{index=4}
    W_inst_base = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v in g_inst.edges():
        w = rng.normal(loc=0.0, scale=weight_scale)
        if abs(w) < 0.1:
            w = 0.1 if w >= 0 else -0.1
        W_inst_base[u, v] = w

    topo_inst = list(nx.topological_sort(g_inst))
    inst_parents = [list(g_inst.predecessors(v)) for v in range(n_nodes)]

    # base lag mats
    A_lags_base = _make_lag_mats(
        n_nodes=n_nodes, tau_max=tau_max, edge_prob=edge_prob,
        rng=rng, weight_scale=weight_scale,
    )

    if intervention_type not in {"hard", "soft_weight", "shift", "noise"}:
        raise ValueError(f"Unknown intervention_type: {intervention_type}")

    blocks: List[pd.DataFrame] = []
    context_graphs: Dict[int, nx.DiGraph] = {}
    interventions: Dict[int, List[str]] = {}

    for c in range(n_contexts):
        targets = rng.choice(n_nodes, size=min(n_intervened_per_context, n_nodes), replace=False).tolist()

        # copy parameters
        W_inst = W_inst_base.copy()
        A_lags = [A.copy() for A in A_lags_base]
        shift = np.zeros(n_nodes, dtype=float)
        noise_vec = np.full(n_nodes, float(noise_scale), dtype=float)
        node_act = [_act("lin") for _ in range(n_nodes)]

        if intervention_type == "hard":
            # zero incoming instantaneous weights into targets
            for t in targets:
                W_inst[:, t] = 0.0
            # zero incoming lagged weights into targets
            for t in targets:
                for A in A_lags:
                    A[:, t] = 0.0

        elif intervention_type == "soft_weight":
            for t in targets:
                W_inst[:, t] *= weight_scale_intervened
                for A in A_lags:
                    A[:, t] *= weight_scale_intervened

        elif intervention_type == "shift":
            for t in targets:
                shift[t] = rng.normal(loc=0.0, scale=shift_scale)

        elif intervention_type == "noise":
            ns = noise_scale_intervened if noise_scale_intervened is not None else (2.0 * float(noise_scale))
            for t in targets:
                noise_vec[t] = ns

        Xc = _simulate_temporal_sem(
            n_samples=n_samples_per_context,
            cols=cols,
            tau_max=tau_max,
            topo_inst=topo_inst,
            inst_parents=inst_parents,
            W_inst=W_inst,
            A_lags=A_lags,
            rng=rng,
            noise_scale=noise_vec,
            shift=shift,
            node_act=node_act,
            burnin=burnin,
        )
        df_c = pd.DataFrame(Xc, columns=cols)
        df_c[context_col] = c
        blocks.append(df_c)

        # context temporal graph
        g_true_c = _temporal_true_graph(cols=cols, tau_max=tau_max, g_inst=g_inst, A_lags=A_lags)
        context_graphs[c] = nx.relabel_nodes(g_true_c, lambda x: x, copy=True)
        interventions[c] = [mapping[t] for t in targets]

    df = pd.concat(blocks, ignore_index=True)

    true_base = _temporal_true_graph(cols=cols, tau_max=tau_max, g_inst=g_inst, A_lags=A_lags_base)

    # for now: use base as target graph (same as your current behavior) :contentReference[oaicite:5]{index=5}
    true_target = true_base
    return df, true_target#df, true_base, true_target, context_graphs, interventions


def sample_multicontext_temporal_nonlinear(
    *,
    n_samples_per_context: int,
    n_nodes: int,
    edge_prob: float,
    n_contexts: int,
    seed: int,
    tau_max: int = 1,
    context_col: str = "context",
    n_intervened_per_context: int = 1,
    nonlinearity: str = "tanh",
    intervention_type: str = "soft_mechanism",  # "hard"|"soft_weight"|"soft_mechanism"|"shift"|"noise"
    weight_scale: float = 0.3,
    noise_scale: float = 0.7,
    weight_scale_intervened: float = 2.0,
    alt_nonlinearity: str = "sin",
    shift_scale: float = 2.0,
    noise_scale_intervened: float | None = None,
    burnin: int | None = None,
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    rng = np.random.default_rng(seed)
    burnin = int(10 * tau_max) if burnin is None else int(burnin)

    cols = [f"X{i}" for i in range(n_nodes)]
    mapping = {i: cols[i] for i in range(n_nodes)}

    g_inst = _random_dag(n_nodes=n_nodes, edge_prob=edge_prob, rng=rng)  # :contentReference[oaicite:6]{index=6}
    W_inst_base = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v in g_inst.edges():
        w = rng.normal(loc=0.0, scale=weight_scale)
        if abs(w) < 0.1:
            w = 0.1 if w >= 0 else -0.1
        W_inst_base[u, v] = w

    topo_inst = list(nx.topological_sort(g_inst))
    inst_parents = [list(g_inst.predecessors(v)) for v in range(n_nodes)]

    A_lags_base = _make_lag_mats(
        n_nodes=n_nodes, tau_max=tau_max, edge_prob=edge_prob,
        rng=rng, weight_scale=weight_scale,
    )

    if intervention_type not in {"hard", "soft_weight", "soft_mechanism", "shift", "noise"}:
        raise ValueError(f"Unknown intervention_type: {intervention_type}")

    f_base = _act(nonlinearity)
    f_alt = _act(alt_nonlinearity)

    blocks: List[pd.DataFrame] = []
    context_graphs: Dict[int, nx.DiGraph] = {}
    interventions: Dict[int, List[str]] = {}

    for c in range(n_contexts):
        targets = rng.choice(n_nodes, size=min(n_intervened_per_context, n_nodes), replace=False).tolist()

        W_inst = W_inst_base.copy()
        A_lags = [A.copy() for A in A_lags_base]
        shift = np.zeros(n_nodes, dtype=float)
        noise_vec = np.full(n_nodes, float(noise_scale), dtype=float)
        node_act = [f_base for _ in range(n_nodes)]

        if intervention_type == "hard":
            for t in targets:
                W_inst[:, t] = 0.0
            for t in targets:
                for A in A_lags:
                    A[:, t] = 0.0

        elif intervention_type == "soft_weight":
            for t in targets:
                W_inst[:, t] *= weight_scale_intervened
                for A in A_lags:
                    A[:, t] *= weight_scale_intervened

        elif intervention_type == "soft_mechanism":
            for t in targets:
                node_act[t] = f_alt

        elif intervention_type == "shift":
            for t in targets:
                shift[t] = rng.normal(loc=0.0, scale=shift_scale)

        elif intervention_type == "noise":
            ns = noise_scale_intervened if noise_scale_intervened is not None else (2.0 * float(noise_scale))
            for t in targets:
                noise_vec[t] = ns

        Xc = _simulate_temporal_sem(
            n_samples=n_samples_per_context,
            cols=cols,
            tau_max=tau_max,
            topo_inst=topo_inst,
            inst_parents=inst_parents,
            W_inst=W_inst,
            A_lags=A_lags,
            rng=rng,
            noise_scale=noise_vec,
            shift=shift,
            node_act=node_act,
            burnin=burnin,
        )
        df_c = pd.DataFrame(Xc, columns=cols)
        df_c[context_col] = c
        blocks.append(df_c)

        g_true_c = _temporal_true_graph(cols=cols, tau_max=tau_max, g_inst=g_inst, A_lags=A_lags)
        context_graphs[c] = nx.relabel_nodes(g_true_c, lambda x: x, copy=True)
        interventions[c] = [mapping[t] for t in targets]

    df = pd.concat(blocks, ignore_index=True)

    true_base = _temporal_true_graph(cols=cols, tau_max=tau_max, g_inst=g_inst, A_lags=A_lags_base)
    true_target = true_base
    return df, true_target #df, true_base, true_target, context_graphs, interventions
