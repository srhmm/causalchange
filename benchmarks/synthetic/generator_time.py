from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpaceTimeSyntheticResult:
    df: pd.DataFrame
    true_wcg: nx.DiGraph
    true_summary_dag: nx.DiGraph
    changepoints: list[int]
    interval_regime_labels: list[int]
    time_regime_labels: list[int]
    context_labels_by_target: dict[str, dict[int, int]]
    regime_labels_by_target: dict[str, dict[int, int]]
    variables: list[str]
    context_col: str
    tau_max: int
    metadata: dict[str, Any]


def sample_spacetime_synthetic(
    *,
    n_datasets: int,
    n_samples: int,
    n_nodes: int,
    edge_prob: float,
    seed: int,
    tau_max: int = 2,
    n_changepoints: int = 2,
    n_regimes: int = 2,
    n_context_clusters: int = 2,
    min_segment_length: int = 30,
    nonlinearity: str = "tanh",
    weight_scale: float = 0.25,
    noise_scale: float = 0.7,
    mechanism_change_fraction: float = 0.5,
    mechanism_shift_scale: float = 0.75,
    context_col: str = "context",
    burnin: int | None = None,
    allow_self_lag: bool = True,
) -> SpaceTimeSyntheticResult:
    """Generate non-stationary multi-dataset time series for SpaceTime benchmarks.

    The graph is invariant across datasets and regimes. Context/regime changes alter
    the structural mechanisms by changing edge weights, rather than by deleting parents.
    """
    if n_datasets < 1:
        raise ValueError("n_datasets must be positive.")
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive.")
    if tau_max < 1:
        raise ValueError("tau_max must be positive.")
    if not 0.0 <= edge_prob <= 1.0:
        raise ValueError("edge_prob must be in [0, 1].")
    if not 0.0 <= mechanism_change_fraction <= 1.0:
        raise ValueError("mechanism_change_fraction must be in [0, 1].")
    if n_changepoints < 0:
        raise ValueError("n_changepoints must be non-negative.")

    n_intervals = n_changepoints + 1
    if n_regimes < 1 or n_regimes > n_intervals:
        raise ValueError("n_regimes must be between 1 and n_changepoints + 1.")
    if n_context_clusters < 1 or n_context_clusters > n_datasets:
        raise ValueError("n_context_clusters must be between 1 and n_datasets.")
    if n_samples < n_intervals * min_segment_length:
        raise ValueError(
            "n_samples is too small for n_changepoints and min_segment_length. "
            f"Need at least {n_intervals * min_segment_length}."
        )

    rng = np.random.default_rng(seed)
    burnin = int(10 * tau_max) if burnin is None else int(burnin)

    variables = [f"X{i}" for i in range(n_nodes)]

    g_inst, W_inst_base = _sample_instantaneous_graph_and_weights(
        n_nodes=n_nodes,
        edge_prob=edge_prob,
        rng=rng,
        weight_scale=weight_scale,
    )
    topo_inst = list(nx.topological_sort(g_inst))
    inst_parents = [list(g_inst.predecessors(v)) for v in range(n_nodes)]

    A_lags_base = _sample_lag_weights(
        n_nodes=n_nodes,
        tau_max=tau_max,
        edge_prob=edge_prob,
        rng=rng,
        weight_scale=weight_scale,
        allow_self_lag=allow_self_lag,
    )

    edge_keys = _edge_keys(W_inst_base, A_lags_base)
    if not edge_keys:
        u = 0
        v = 0 if n_nodes == 1 else 1
        A_lags_base[0][u, v] = _nonzero_normal(rng, weight_scale)
        edge_keys = _edge_keys(W_inst_base, A_lags_base)

    n_changed_edges = int(np.ceil(mechanism_change_fraction * len(edge_keys)))
    n_changed_edges = min(max(n_changed_edges, 0), len(edge_keys))

    context_shift_edges = _choose_edges(edge_keys, n_changed_edges, rng)
    regime_shift_edges = _choose_edges(edge_keys, n_changed_edges, rng)

    segment_lengths = _segment_lengths(
        n_samples=n_samples,
        n_intervals=n_intervals,
        min_segment_length=min_segment_length,
        rng=rng,
    )
    changepoints = np.cumsum(segment_lengths)[:-1].astype(int).tolist()

    # Cyclic labels make recurrent regimes possible, e.g. [0, 1, 0].
    interval_regime_labels = [i % n_regimes for i in range(n_intervals)]
    time_regime_labels = _expand_interval_labels(segment_lengths, interval_regime_labels)

    dataset_context_labels = _balanced_labels(
        n_items=n_datasets,
        n_clusters=n_context_clusters,
        rng=rng,
    )

    context_factors = _sample_shift_factors(
        labels=range(n_context_clusters),
        edge_keys=context_shift_edges,
        rng=rng,
        mechanism_shift_scale=mechanism_shift_scale,
    )
    regime_factors = _sample_shift_factors(
        labels=range(n_regimes),
        edge_keys=regime_shift_edges,
        rng=rng,
        mechanism_shift_scale=mechanism_shift_scale,
    )

    cell_params = {
        (context_label, regime_label): _build_cell_params(
            W_inst_base=W_inst_base,
            A_lags_base=A_lags_base,
            context_label=context_label,
            regime_label=regime_label,
            context_factors=context_factors,
            regime_factors=regime_factors,
        )
        for context_label in range(n_context_clusters)
        for regime_label in range(n_regimes)
    }

    act = _activation(nonlinearity)
    blocks: list[pd.DataFrame] = []

    for dataset_id in range(n_datasets):
        context_label = dataset_context_labels[dataset_id]
        X = _simulate_dataset(
            n_samples=n_samples,
            n_nodes=n_nodes,
            tau_max=tau_max,
            topo_inst=topo_inst,
            inst_parents=inst_parents,
            cell_params=cell_params,
            context_label=context_label,
            time_regime_labels=time_regime_labels,
            rng=rng,
            noise_scale=float(noise_scale),
            activation=act,
            burnin=burnin,
        )

        df_d = pd.DataFrame(X, columns=variables)

        # This is the observed dataset id D. The latent context cluster is stored
        # separately in context_labels_by_target / metadata.
        df_d[context_col] = dataset_id
        blocks.append(df_d)

    df = pd.concat(blocks, ignore_index=True)

    true_wcg = _window_causal_graph(variables, W_inst_base, A_lags_base)
    true_summary_dag = _summary_graph(variables, true_wcg)

    targets_with_context_shift = _targets_of_edges(context_shift_edges, variables)
    targets_with_regime_shift = _targets_of_edges(regime_shift_edges, variables)

    context_labels_by_target = {
        target: (
            {dataset_id: int(dataset_context_labels[dataset_id]) for dataset_id in range(n_datasets)}
            if target in targets_with_context_shift
            else {dataset_id: 0 for dataset_id in range(n_datasets)}
        )
        for target in variables
    }

    regime_labels_by_target = {
        target: (
            {interval_id: int(label) for interval_id, label in enumerate(interval_regime_labels)}
            if target in targets_with_regime_shift
            else {interval_id: 0 for interval_id in range(n_intervals)}
        )
        for target in variables
    }

    return SpaceTimeSyntheticResult(
        df=df,
        true_wcg=true_wcg,
        true_summary_dag=true_summary_dag,
        changepoints=changepoints,
        interval_regime_labels=[int(x) for x in interval_regime_labels],
        time_regime_labels=[int(x) for x in time_regime_labels],
        context_labels_by_target=context_labels_by_target,
        regime_labels_by_target=regime_labels_by_target,
        variables=variables,
        context_col=context_col,
        tau_max=tau_max,
        metadata={
            "dataset_context_labels": {int(k): int(v) for k, v in dataset_context_labels.items()},
            "segment_lengths": [int(x) for x in segment_lengths],
            "context_shift_edges": list(context_shift_edges),
            "regime_shift_edges": list(regime_shift_edges),
            "n_context_clusters": n_context_clusters,
            "n_regimes": n_regimes,
            "nonlinearity": nonlinearity,
        },
    )


def _sample_instantaneous_graph_and_weights(
    *,
    n_nodes: int,
    edge_prob: float,
    rng: np.random.Generator,
    weight_scale: float,
) -> tuple[nx.DiGraph, np.ndarray]:
    order = rng.permutation(n_nodes).tolist()
    pos = {node: i for i, node in enumerate(order)}

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))

    W = np.zeros((n_nodes, n_nodes), dtype=float)

    for u in range(n_nodes):
        for v in range(n_nodes):
            if u == v or pos[u] >= pos[v]:
                continue
            if rng.random() < edge_prob:
                graph.add_edge(u, v)
                W[u, v] = _nonzero_normal(rng, weight_scale)

    return graph, W


def _sample_lag_weights(
    *,
    n_nodes: int,
    tau_max: int,
    edge_prob: float,
    rng: np.random.Generator,
    weight_scale: float,
    allow_self_lag: bool,
) -> list[np.ndarray]:
    mats: list[np.ndarray] = []

    for _lag in range(1, tau_max + 1):
        A = np.zeros((n_nodes, n_nodes), dtype=float)

        for u in range(n_nodes):
            for v in range(n_nodes):
                if u == v and not allow_self_lag:
                    continue
                if rng.random() < edge_prob:
                    A[u, v] = _nonzero_normal(rng, weight_scale)

        mats.append(A)

    return mats


def _nonzero_normal(rng: np.random.Generator, scale: float) -> float:
    value = float(rng.normal(loc=0.0, scale=scale))
    if abs(value) < 0.05:
        value = 0.05 if value >= 0 else -0.05
    return value


def _edge_keys(
    W_inst: np.ndarray,
    A_lags: list[np.ndarray],
) -> list[tuple[str, int, int, int]]:
    keys: list[tuple[str, int, int, int]] = []

    for u, v in zip(*np.nonzero(W_inst), strict=False):
        keys.append(("inst", 0, int(u), int(v)))

    for lag, A in enumerate(A_lags, start=1):
        for u, v in zip(*np.nonzero(A), strict=False):
            keys.append(("lag", int(lag), int(u), int(v)))

    return keys


def _choose_edges(
    edge_keys: list[tuple[str, int, int, int]],
    n_edges: int,
    rng: np.random.Generator,
) -> tuple[tuple[str, int, int, int], ...]:
    if n_edges == 0:
        return tuple()

    idx = rng.choice(len(edge_keys), size=n_edges, replace=False)
    return tuple(edge_keys[int(i)] for i in idx)


def _segment_lengths(
    *,
    n_samples: int,
    n_intervals: int,
    min_segment_length: int,
    rng: np.random.Generator,
) -> list[int]:
    remaining = n_samples - n_intervals * min_segment_length

    if remaining > 0:
        extras = rng.multinomial(remaining, np.ones(n_intervals) / n_intervals)
    else:
        extras = np.zeros(n_intervals, dtype=int)

    return (min_segment_length + extras).astype(int).tolist()


def _expand_interval_labels(
    segment_lengths: list[int],
    interval_labels: list[int],
) -> list[int]:
    labels: list[int] = []

    for length, label in zip(segment_lengths, interval_labels, strict=True):
        labels.extend([int(label)] * int(length))

    return labels


def _balanced_labels(
    *,
    n_items: int,
    n_clusters: int,
    rng: np.random.Generator,
) -> dict[int, int]:
    labels = np.arange(n_items, dtype=int) % int(n_clusters)
    rng.shuffle(labels)

    return {item: int(label) for item, label in enumerate(labels.tolist())}


def _sample_shift_factors(
    *,
    labels: range,
    edge_keys: tuple[tuple[str, int, int, int], ...],
    rng: np.random.Generator,
    mechanism_shift_scale: float,
) -> dict[tuple[int, tuple[str, int, int, int]], float]:
    factors: dict[tuple[int, tuple[str, int, int, int]], float] = {}

    for label in labels:
        for edge_key in edge_keys:
            if int(label) == 0:
                factors[(int(label), edge_key)] = 1.0
            else:
                factor = float(rng.normal(loc=1.0, scale=mechanism_shift_scale))
                if abs(factor) < 0.2:
                    factor = 0.2 if factor >= 0 else -0.2
                factors[(int(label), edge_key)] = factor

    return factors


def _build_cell_params(
    *,
    W_inst_base: np.ndarray,
    A_lags_base: list[np.ndarray],
    context_label: int,
    regime_label: int,
    context_factors: dict[tuple[int, tuple[str, int, int, int]], float],
    regime_factors: dict[tuple[int, tuple[str, int, int, int]], float],
) -> tuple[np.ndarray, list[np.ndarray]]:
    W = W_inst_base.copy()
    A_lags = [A.copy() for A in A_lags_base]

    all_shift_edges = {key for _, key in context_factors} | {key for _, key in regime_factors}

    for edge_key in all_shift_edges:
        kind, lag, u, v = edge_key

        factor = context_factors.get((int(context_label), edge_key), 1.0)
        factor *= regime_factors.get((int(regime_label), edge_key), 1.0)

        if kind == "inst":
            W[u, v] *= factor
        elif kind == "lag":
            A_lags[lag - 1][u, v] *= factor
        else:
            raise ValueError(f"Unknown edge kind: {kind!r}")

    return W, A_lags


def _activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name == "lin":
        return lambda x: x
    if name == "tanh":
        return np.tanh
    if name == "sin":
        return np.sin
    if name == "relu":
        return lambda x: np.maximum(x, 0.0)

    raise ValueError(f"Unknown nonlinearity: {name!r}")


def _simulate_dataset(
    *,
    n_samples: int,
    n_nodes: int,
    tau_max: int,
    topo_inst: list[int],
    inst_parents: list[list[int]],
    cell_params: dict[tuple[int, int], tuple[np.ndarray, list[np.ndarray]]],
    context_label: int,
    time_regime_labels: list[int],
    rng: np.random.Generator,
    noise_scale: float,
    activation: Callable[[np.ndarray], np.ndarray],
    burnin: int,
) -> np.ndarray:
    total = n_samples + burnin + tau_max

    X = np.zeros((total, n_nodes), dtype=float)
    X[:tau_max, :] = rng.normal(
        loc=0.0,
        scale=noise_scale,
        size=(tau_max, n_nodes),
    )

    eps = rng.normal(loc=0.0, scale=noise_scale, size=(total, n_nodes))

    for t in range(tau_max, total):
        obs_t = t - tau_max - burnin
        regime_label = time_regime_labels[obs_t] if obs_t >= 0 else time_regime_labels[0]

        W_inst, A_lags = cell_params[(int(context_label), int(regime_label))]

        lag_contrib = np.zeros(n_nodes, dtype=float)
        for lag, A in enumerate(A_lags, start=1):
            lag_contrib += X[t - lag, :] @ A

        x_t = np.zeros(n_nodes, dtype=float)

        for v in topo_inst:
            inst = 0.0
            parents = inst_parents[v]

            if parents:
                inst = float(x_t[parents] @ W_inst[parents, v])

            x_t[v] = float(activation(np.array(lag_contrib[v] + inst)) + eps[t, v])

        X[t, :] = x_t

    return X[tau_max + burnin : tau_max + burnin + n_samples, :]


def _window_causal_graph(
    variables: list[str],
    W_inst: np.ndarray,
    A_lags: list[np.ndarray],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    tau_max = len(A_lags)

    for variable in variables:
        for lag in range(tau_max + 1):
            graph.add_node((variable, lag))

    for u, v in zip(*np.nonzero(W_inst), strict=False):
        graph.add_edge((variables[int(u)], 0), (variables[int(v)], 0))

    for lag, A in enumerate(A_lags, start=1):
        for u, v in zip(*np.nonzero(A), strict=False):
            graph.add_edge((variables[int(u)], lag), (variables[int(v)], 0))

    return graph


def _summary_graph(variables: list[str], wcg: nx.DiGraph) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(variables)

    for cause, effect in wcg.edges():
        cause_var, _cause_lag = cause
        effect_var, _effect_lag = effect

        if cause_var != effect_var:
            graph.add_edge(cause_var, effect_var)

    return graph


def _targets_of_edges(
    edge_keys: tuple[tuple[str, int, int, int], ...],
    variables: list[str],
) -> set[str]:
    return {variables[v] for _kind, _lag, _u, v in edge_keys}
