from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pandas as pd

from benchmarks.synthetic.generator_time import sample_spacetime_synthetic
from causalchange.config.benchmark_config import (
    MultiDataConfig,
    MultiTemporalDataConfig,
    SingleDataConfig,
    SingleTemporalDataConfig,
)


def sample_single_continuous(config: SingleDataConfig):
    sampling_fun = sample_linear_gaussian if config.nonlinearity == "lin" else sample_nonlinear_additive
    return sampling_fun(
        nonlinearity=config.nonlinearity,
        n_samples=config.n_samples,
        n_nodes=config.n_nodes,
        edge_prob=config.edge_prob,
        seed=config.seed,
        weight_scale=config.weight_scale,
        noise_scale=config.noise_scale,
    )


def sample_multi_continuous(config: MultiDataConfig):
    sampling_fun = (
        sample_multicontext_linear_gaussian_interventional
        if config.nonlinearity == "lin"
        else sample_multicontext_nonlinear_additive_interventional
    )

    return sampling_fun(
        nonlinearity=config.nonlinearity,
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


def sample_single_temporal(config: SingleTemporalDataConfig):
    context_col = "__context__"

    res = sample_spacetime_synthetic(
        n_datasets=1,
        n_samples=config.n_samples,
        n_nodes=config.n_nodes,
        edge_prob=config.edge_prob,
        seed=config.seed,
        tau_max=config.tau_max,
        n_changepoints=config.n_changepoints,
        n_regimes=config.n_regimes,
        n_context_clusters=1,
        min_segment_length=config.min_segment_length,
        nonlinearity=config.nonlinearity,
        weight_scale=config.weight_scale,
        noise_scale=config.noise_scale,
        mechanism_change_fraction=config.mechanism_change_fraction,
        mechanism_shift_scale=config.mechanism_shift_scale,
        context_col=context_col,
        burnin=config.burnin,
        allow_self_lag=config.allow_self_lag,
    )

    df = res.df.drop(columns=[context_col])
    return replace(res, df=df)


def sample_multi_temporal(config: MultiTemporalDataConfig):
    return sample_spacetime_synthetic(
        n_datasets=config.n_datasets or config.n_contexts,
        n_samples=config.n_samples_per_context,
        n_nodes=config.n_nodes,
        edge_prob=config.edge_prob,
        seed=config.seed,
        tau_max=config.tau_max,
        n_changepoints=config.n_changepoints,
        n_regimes=config.n_regimes,
        n_context_clusters=config.n_context_clusters,
        min_segment_length=config.min_segment_length,
        nonlinearity=config.nonlinearity,
        weight_scale=config.weight_scale,
        noise_scale=config.noise_scale,
        mechanism_change_fraction=config.mechanism_change_fraction,
        mechanism_shift_scale=config.mechanism_shift_scale,
        context_col=config.context_col,
        burnin=config.burnin,
        allow_self_lag=config.allow_self_lag,
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
) -> tuple[pd.DataFrame, nx.DiGraph]:
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

        def f(x):
            return np.maximum(x, 0.0)

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
    nonlinearity: str = "",
) -> tuple[pd.DataFrame, nx.DiGraph]:
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


def _cut_incoming_edges(g: nx.DiGraph, targets: list[int]) -> nx.DiGraph:
    gc = g.copy()
    for t in targets:
        for p in list(gc.predecessors(t)):
            gc.remove_edge(p, t)
    return gc


def _intersect_edges(graphs: list[nx.DiGraph]) -> nx.DiGraph:
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

        def f(x):
            return np.maximum(x, 0.0)

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
    nonlinearity="",
) -> tuple[pd.DataFrame, nx.DiGraph]:
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
    context_graphs: dict[int, nx.DiGraph] = {}
    interventions: dict[int, list[str]] = {}
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
    return df, true_target  # df, true_base, true_target, context_graphs, interventions


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
) -> tuple[pd.DataFrame, nx.DiGraph]:
    rng = np.random.default_rng(seed)
    g_base = _random_dag(n_nodes=n_nodes, edge_prob=edge_prob, rng=rng)

    W_base = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v in g_base.edges():
        w = rng.normal(loc=0.0, scale=weight_scale)
        if abs(w) < 0.1:
            w = 0.1 if w >= 0 else -0.1
        W_base[u, v] = w

    if intervention_type not in {
        "hard",
        "soft_weight",
        "soft_mechanism",
        "shift",
        "noise",
    }:
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
    context_graphs: dict[int, nx.DiGraph] = {}
    interventions: dict[int, list[str]] = {}
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
    return df, true_target  # df, true_base, true_target, context_graphs, interventions


def _true_target_graph(
    true_base: nx.DiGraph,
    context_graphs: dict[int, nx.DiGraph],
    intervention_type: str,
) -> nx.DiGraph:
    return true_base


# if intervention_type == "hard":
#    return _intersect_edges(list(context_graphs.values()))
# return true_base


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
