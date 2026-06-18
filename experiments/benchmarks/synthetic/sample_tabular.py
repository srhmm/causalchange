from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pandas as pd

from causalchange.config.benchmark_config import (
    MixedDataConfig,
    MultiDataConfig,
    MultiTemporalDataConfig,
    SingleDataConfig,
    SingleTemporalDataConfig,
)
from experiments.benchmarks.synthetic.sample import MixedSyntheticResult, MultiContextSyntheticResult
from experiments.benchmarks.synthetic.sample_time import sample_spacetime_synthetic


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

    kwargs = dict(
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

    if config.nonlinearity != "lin":
        kwargs["alt_nonlinearity"] = config.alt_nonlinearity or ("sin" if config.nonlinearity != "sin" else "tanh")

    return sampling_fun(**kwargs)


def sample_mixed_continuous(config: MixedDataConfig):
    return sample_latent_mixed_continuous(
        nonlinearity=config.nonlinearity,
        alt_nonlinearity=config.alt_nonlinearity,
        n_samples_per_mechanism=config.n_samples_per_mechanism,
        n_nodes=config.n_nodes,
        edge_prob=config.edge_prob,
        n_mechanisms=config.n_mechanisms,
        n_mixed_variables=config.n_mixed_variables,
        cluster_mode=config.cluster_mode,
        seed=config.seed,
        weight_scale=config.weight_scale,
        noise_scale=config.noise_scale,
        mechanism_change=config.mechanism_change,
        weight_scale_intervened=config.weight_scale_intervened,
        shift_scale=config.shift_scale,
        noise_scale_intervened=config.noise_scale_intervened,
    )


def sample_latent_mixed_continuous(
    *,
    n_samples_per_mechanism: int,
    n_nodes: int,
    edge_prob: float,
    n_mechanisms: int,
    n_mixed_variables: int,
    cluster_mode: str,
    seed: int,
    weight_scale: float = 2.0,
    noise_scale: float = 0.7,
    nonlinearity: str = "tanh",
    alt_nonlinearity: str | None = None,
    mechanism_change: str = "soft-weight",
    weight_scale_intervened: float = 2.0,
    shift_scale: float = 2.0,
    noise_scale_intervened: float | None = None,
) -> MixedSyntheticResult:
    """Generate data from a mixed SCM without observed mixture labels.

    One invariant DAG is sampled. A subset of variables has multiple latent
    causal mechanisms.

    In global mode, all mixed variables share one latent row clustering.
    In local mode, each mixed variable has its own latent row clustering.
    """

    if cluster_mode not in {"global", "local"}:
        raise ValueError(f"Unknown cluster_mode: {cluster_mode}")

    if mechanism_change not in {"soft-weight", "soft-mechanism", "shift", "noise"}:
        raise ValueError(f"Unknown mechanism_change: {mechanism_change}")

    if mechanism_change == "soft-mechanism" and alt_nonlinearity is None:
        alt_nonlinearity = "sin" if nonlinearity != "sin" else "tanh"

    rng = np.random.default_rng(seed)

    g_base = _random_dag(n_nodes=n_nodes, edge_prob=edge_prob, rng=rng)
    topo = list(nx.topological_sort(g_base))

    W_base = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v in g_base.edges():
        w = rng.normal(loc=0.0, scale=weight_scale)
        if abs(w) < 0.1:
            w = 0.1 if w >= 0 else -0.1
        W_base[u, v] = w

    n_mixed = min(n_mixed_variables, n_nodes)
    mixed_nodes = set(rng.choice(n_nodes, size=n_mixed, replace=False).tolist())

    n_samples = n_samples_per_mechanism * n_mechanisms

    f_base = _act(nonlinearity)
    f_alt = _act(alt_nonlinearity) if alt_nonlinearity is not None else f_base

    labels_by_node = _sample_mixed_labels(
        n_samples=n_samples,
        n_nodes=n_nodes,
        n_mechanisms=n_mechanisms,
        mixed_nodes=mixed_nodes,
        cluster_mode=cluster_mode,
        rng=rng,
    )

    mechanism_params = _sample_node_mixture_parameters(
        g_base=g_base,
        W_base=W_base,
        mixed_nodes=mixed_nodes,
        n_mechanisms=n_mechanisms,
        mechanism_change=mechanism_change,
        weight_scale_intervened=weight_scale_intervened,
        shift_scale=shift_scale,
        noise_scale=noise_scale,
        noise_scale_intervened=noise_scale_intervened,
        f_base=f_base,
        f_alt=f_alt,
        rng=rng,
    )

    X = np.zeros((n_samples, n_nodes), dtype=float)
    eps = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_nodes))

    for v in topo:
        parents = list(g_base.predecessors(v))
        labels_v = labels_by_node[v]

        for z in range(n_mechanisms):
            idx = np.flatnonzero(labels_v == z)
            if len(idx) == 0:
                continue

            params = mechanism_params[v][z]
            weights_v = params["weights"]
            shift_v = params["shift"]
            noise_v = params["noise_scale"]
            f_v = params["function"]

            if parents:
                lin = X[np.ix_(idx, parents)] @ weights_v[parents]
                X[idx, v] = f_v(lin) + shift_v + noise_v * eps[idx, v]
            else:
                X[idx, v] = shift_v + noise_v * eps[idx, v]

    cols = [f"X{i}" for i in range(n_nodes)]
    df = pd.DataFrame(X, columns=cols)

    perm = rng.permutation(len(df))
    df = df.iloc[perm].reset_index(drop=True)

    true_g = nx.relabel_nodes(g_base, {i: cols[i] for i in range(n_nodes)}, copy=True)

    labels_by_target = {cols[node]: labels_by_node[node][perm].astype(int).tolist() for node in range(n_nodes)}

    mixed_targets = [cols[node] for node in sorted(mixed_nodes)]

    return MixedSyntheticResult(
        df=df,
        true_summary_dag=true_g,
        labels_by_target=labels_by_target,
        mixed_targets=mixed_targets,
        cluster_mode=cluster_mode,
        n_mechanisms=int(n_mechanisms),
        metadata={
            "mixed_nodes": [int(node) for node in sorted(mixed_nodes)],
            "mixed_targets": mixed_targets,
            "cluster_mode": cluster_mode,
            "n_mechanisms": int(n_mechanisms),
            "n_mixed_variables": int(n_mixed_variables),
            "mechanism_change": mechanism_change,
        },
    )


def _sample_mixed_labels(
    *,
    n_samples: int,
    n_nodes: int,
    n_mechanisms: int,
    mixed_nodes: set[int],
    cluster_mode: str,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    labels_by_node = {node: np.zeros(n_samples, dtype=int) for node in range(n_nodes)}

    if cluster_mode == "global":
        global_labels = _balanced_labels(
            n_samples=n_samples,
            n_clusters=n_mechanisms,
            rng=rng,
        )

        for node in mixed_nodes:
            labels_by_node[node] = global_labels.copy()

    elif cluster_mode == "local":
        for node in mixed_nodes:
            labels_by_node[node] = _balanced_labels(
                n_samples=n_samples,
                n_clusters=n_mechanisms,
                rng=rng,
            )

    else:
        raise ValueError(f"Unknown cluster_mode: {cluster_mode}")

    return labels_by_node


def _balanced_labels(
    *,
    n_samples: int,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    reps = int(np.ceil(n_samples / n_clusters))
    labels = np.tile(np.arange(n_clusters, dtype=int), reps)[:n_samples]
    rng.shuffle(labels)
    return labels


def _sample_node_mixture_parameters(
    *,
    g_base: nx.DiGraph,
    W_base: np.ndarray,
    mixed_nodes: set[int],
    n_mechanisms: int,
    mechanism_change: str,
    weight_scale_intervened: float,
    shift_scale: float,
    noise_scale: float,
    noise_scale_intervened: float | None,
    f_base,
    f_alt,
    rng: np.random.Generator,
) -> dict[int, list[dict[str, object]]]:
    n_nodes = W_base.shape[0]
    mechanism_params: dict[int, list[dict[str, object]]] = {}

    for node in range(n_nodes):
        node_params: list[dict[str, object]] = []

        for z in range(n_mechanisms):
            weights = W_base[:, node].copy()
            shift = 0.0
            noise = noise_scale
            function = f_base

            if node in mixed_nodes and z > 0:
                if mechanism_change == "soft-weight":
                    for parent in list(g_base.predecessors(node)):
                        factor = rng.normal(loc=weight_scale_intervened, scale=0.25)
                        if abs(factor) < 0.2:
                            factor = 0.2 if factor >= 0 else -0.2
                        weights[parent] = weights[parent] * factor

                elif mechanism_change == "soft-mechanism":
                    function = f_alt

                elif mechanism_change == "shift":
                    shift = float(rng.normal(loc=0.0, scale=shift_scale))

                elif mechanism_change == "noise":
                    noise = float(noise_scale_intervened) if noise_scale_intervened is not None else 2.0 * noise_scale

                else:
                    raise ValueError(f"Unknown mechanism_change: {mechanism_change}")

            node_params.append(
                {
                    "weights": weights,
                    "shift": shift,
                    "noise_scale": noise,
                    "function": function,
                }
            )

        mechanism_params[node] = node_params

    return mechanism_params


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
    intervention_type: str = "soft-weight",  # "hard"|"soft-weight"|"shift"|"noise"
    weight_scale_intervened: float = 2.0,
    shift_scale: float = 2.0,
    noise_scale_intervened: float | None = None,
    nonlinearity: str = "",
) -> MultiContextSyntheticResult:
    rng = np.random.default_rng(seed)
    g_base = _random_dag(n_nodes=n_nodes, edge_prob=edge_prob, rng=rng)

    W_base = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v in g_base.edges():
        w = rng.normal(loc=0.0, scale=weight_scale)
        if abs(w) < 0.1:
            w = 0.1 if w >= 0 else -0.1
        W_base[u, v] = w

    if intervention_type not in {"hard", "soft-weight", "shift", "noise"}:
        raise ValueError(f"Unknown intervention_type: {intervention_type}")

    contexts = list(range(n_contexts))
    context_graphs: dict[int, nx.DiGraph] = {}
    interventions: dict[int, list[str]] = {}
    mechanism_signatures_by_node: dict[int, dict[int, tuple[object, ...]]] = {node: {} for node in range(n_nodes)}
    blocks = []

    cols = [f"X{i}" for i in range(n_nodes)]
    mapping = {i: cols[i] for i in range(n_nodes)}

    for c in contexts:
        targets = rng.choice(
            n_nodes,
            size=min(n_intervened_per_context, n_nodes),
            replace=False,
        ).tolist()

        if intervention_type == "hard":
            g_c = _cut_incoming_edges(g_base, targets)
        else:
            g_c = g_base

        W_c = W_base.copy()
        shift = np.zeros(n_nodes, dtype=float)
        noise_vec = np.full(n_nodes, noise_scale, dtype=float)

        if intervention_type == "soft-weight":
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

        for node in range(n_nodes):
            mechanism_signatures_by_node[node][c] = _mechanism_signature(
                g_c=g_c,
                W_c=W_c,
                shift=shift,
                noise_vec=noise_vec,
                node=node,
                function_name="lin",
            )

        topo = list(nx.topological_sort(g_c))
        X = np.zeros((n_samples_per_context, n_nodes), dtype=float)

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

    context_labels_by_target = {
        cols[node]: _compact_signatures(mechanism_signatures_by_node[node]) for node in range(n_nodes)
    }

    return MultiContextSyntheticResult(
        df=df,
        true_summary_dag=true_target,
        context_labels_by_target=context_labels_by_target,
        context_col=context_col,
        variables=cols,
        metadata={
            "setting": "multi",
            "nonlinearity": "lin",
            "intervention_type": intervention_type,
            "n_contexts": int(n_contexts),
            "n_samples_per_context": int(n_samples_per_context),
            "n_intervened_per_context": int(n_intervened_per_context),
            "interventions": interventions,
        },
    )


def _round_signature_float(x: float, ndigits: int = 10) -> float:
    return round(float(x), ndigits)


def _compact_signatures(signatures: dict[int, tuple[object, ...]]) -> dict[int, int]:
    label_by_signature: dict[tuple[object, ...], int] = {}
    labels: dict[int, int] = {}

    for context_id, signature in signatures.items():
        if signature not in label_by_signature:
            label_by_signature[signature] = len(label_by_signature)
        labels[int(context_id)] = int(label_by_signature[signature])

    return labels


def _mechanism_signature(
    *,
    g_c: nx.DiGraph,
    W_c: np.ndarray,
    shift: np.ndarray,
    noise_vec: np.ndarray,
    node: int,
    function_name: str,
) -> tuple[object, ...]:
    parents = tuple(sorted(g_c.predecessors(node)))
    weights = tuple(_round_signature_float(W_c[p, node]) for p in parents)

    return (
        parents,
        weights,
        _round_signature_float(shift[node]),
        _round_signature_float(noise_vec[node]),
        str(function_name),
    )


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
    intervention_type: str = "soft-mechanism",  # "hard"|"soft-weight"|"soft-mechanism"|"shift"|"noise"
    weight_scale_intervened: float = 2.0,
    alt_nonlinearity: str = "sin",
    shift_scale: float = 2.0,
    noise_scale_intervened: float | None = None,
) -> MultiContextSyntheticResult:
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
        "soft-weight",
        "soft-mechanism",
        "shift",
        "noise",
    }:
        raise ValueError(f"Unknown intervention_type: {intervention_type}")

    def _act_local(name: str):
        if name == "tanh":
            return np.tanh
        if name == "sin":
            return np.sin
        if name == "relu":
            return lambda x: np.maximum(x, 0.0)
        if name == "lin":
            return lambda x: x
        raise ValueError(f"Unknown nonlinearity: {name}")

    f_base = _act_local(nonlinearity)
    f_alt = _act_local(alt_nonlinearity)

    contexts = list(range(n_contexts))
    context_graphs: dict[int, nx.DiGraph] = {}
    interventions: dict[int, list[str]] = {}
    mechanism_signatures_by_node: dict[int, dict[int, tuple[object, ...]]] = {node: {} for node in range(n_nodes)}
    blocks = []

    cols = [f"X{i}" for i in range(n_nodes)]
    mapping = {i: cols[i] for i in range(n_nodes)}

    for c in contexts:
        targets = rng.choice(
            n_nodes,
            size=min(n_intervened_per_context, n_nodes),
            replace=False,
        ).tolist()

        if intervention_type == "hard":
            g_c = _cut_incoming_edges(g_base, targets)
        else:
            g_c = g_base

        W_c = W_base.copy()
        shift = np.zeros(n_nodes, dtype=float)
        noise_vec = np.full(n_nodes, noise_scale, dtype=float)

        node_f = [f_base for _ in range(n_nodes)]
        node_f_name = [nonlinearity for _ in range(n_nodes)]

        if intervention_type == "soft-weight":
            for t in targets:
                for p in list(g_base.predecessors(t)):
                    W_c[p, t] = W_c[p, t] * weight_scale_intervened

        elif intervention_type == "soft-mechanism":
            for t in targets:
                node_f[t] = f_alt
                node_f_name[t] = alt_nonlinearity

        elif intervention_type == "shift":
            for t in targets:
                shift[t] = rng.normal(loc=0.0, scale=shift_scale)

        elif intervention_type == "noise":
            ns = noise_scale_intervened if noise_scale_intervened is not None else (2.0 * noise_scale)
            for t in targets:
                noise_vec[t] = ns

        for node in range(n_nodes):
            mechanism_signatures_by_node[node][c] = _mechanism_signature(
                g_c=g_c,
                W_c=W_c,
                shift=shift,
                noise_vec=noise_vec,
                node=node,
                function_name=node_f_name[node],
            )

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

    context_labels_by_target = {
        cols[node]: _compact_signatures(mechanism_signatures_by_node[node]) for node in range(n_nodes)
    }

    return MultiContextSyntheticResult(
        df=df,
        true_summary_dag=true_target,
        context_labels_by_target=context_labels_by_target,
        context_col=context_col,
        variables=cols,
        metadata={
            "setting": "multi",
            "nonlinearity": nonlinearity,
            "alt_nonlinearity": alt_nonlinearity,
            "intervention_type": intervention_type,
            "n_contexts": int(n_contexts),
            "n_samples_per_context": int(n_samples_per_context),
            "n_intervened_per_context": int(n_intervened_per_context),
            "interventions": interventions,
        },
    )


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
