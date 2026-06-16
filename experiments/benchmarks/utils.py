import dataclasses
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import networkx as nx
import numpy as np

from causalchange import CausalChange
from causalchange.core.results import GridCell
from experiments.benchmarks.synthetic.metrics_time import compute_partition_metrics


def _pgmpy_graph_to_nx(dag: Any) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from([str(n) for n in dag.nodes()])
    g.add_edges_from([(str(u), str(v)) for (u, v) in dag.edges()])
    return g


def mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"))
    m = sum(xs) / n
    if n == 1:
        return (m, 0.0)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return (m, math.sqrt(var))


def flatten_dict(d: dict[str, Any], parent: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep=sep))
        else:
            out[key] = v
    return out


def _freeze(v: Any) -> Any:
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(val)) for k, val in v.items()))
    if isinstance(v, set):
        return tuple(sorted(_freeze(x) for x in v))
    if isinstance(v, list) or isinstance(v, tuple):
        return tuple(_freeze(x) for x in v)
    return v


def config_group_key(cfg) -> tuple[tuple[str, Any], ...]:
    d = cfg.model_dump()
    d.get("data", {}).pop("seed", None)
    frozen = _freeze(d)
    return frozen


@dataclass(frozen=True)
class SummaryRow:
    bench: str
    metric: str
    mean: float
    std: float
    n: int
    config: dict[str, Any]


def file_name_from_cfg(cfg) -> str:
    d = cfg.model_dump()
    data = d["data"]
    algo = d["algo"]

    parts = [
        data["setting"],
        algo["name"],
        data["nonlinearity"],
        f"nn-{data.get('n_nodes')}",
        f"p-{data.get('edge_prob')}",
    ]

    if data["setting"] == "multi":
        parts.append(f"nc-{data.get('n_contexts')}")
        parts.append(f"iv-{data.get('intervention_type')}")
        parts.append(f"niv-{data.get('n_intervened_per_context')}")
        parts.append(f"nsc-{data.get('n_samples_per_context')}")
    elif data["setting"] == "time-contexts":
        parts.append(f"nd-{data.get('n_datasets') or data.get('n_contexts')}")
        parts.append(f"nsc-{data.get('n_samples_per_context')}")
        parts.append(f"tau-{data.get('tau_max')}")
        parts.append(f"cp-{data.get('n_changepoints')}")
        parts.append(f"reg-{data.get('n_regimes')}")
        parts.append(f"ctx-{data.get('n_context_clusters')}")
    elif data["setting"] == "time":
        parts.append(f"ns-{data.get('n_samples')}")
        parts.append(f"tau-{data.get('tau_max')}")
        parts.append(f"cp-{data.get('n_changepoints')}")
        parts.append(f"reg-{data.get('n_regimes')}")
    else:
        parts.append(f"ns-{data.get('n_samples')}")

    return "_".join(str(p) for p in parts)


def bench_name_from_cfg(cfg) -> str:
    data = cfg.data
    algo = cfg.algo

    parts = [
        algo.name,
        data.setting,
        f"fun-{data.nonlinearity}",
        f"score-{algo.score_type}",
        f"nn-{data.n_nodes}",
        f"p-{data.edge_prob}",
    ]

    if hasattr(data, "n_samples"):
        parts.append(f"n-{data.n_samples}")

    if hasattr(data, "n_samples_per_context"):
        parts.append(f"nctx-{data.n_samples_per_context}")

    if hasattr(data, "n_contexts"):
        parts.append(f"ctx-{data.n_contexts}")

    if hasattr(data, "tau_max"):
        parts.append(f"tau-{data.tau_max}")

    if hasattr(data, "n_changepoints"):
        parts.append(f"cp-{data.n_changepoints}")

    return "_".join(str(p) for p in parts)


def summarize_groups(
    groups: dict[tuple[tuple[str, Any], ...], dict[str, Any]],
) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    for _, payload in groups.items():
        config_example = payload["config_example"]
        bench = payload["bench"]
        metrics_map: dict[str, list[float]] = payload["metrics"]

        for metric, values in metrics_map.items():
            m, s = mean_std(values)
            rows.append(
                SummaryRow(
                    bench=bench,
                    metric=metric,
                    mean=m,
                    std=s,
                    n=len(values),
                    config=config_example,
                )
            )

    rows.sort(key=lambda r: (r.bench, r.metric))
    return rows


def to_json_safe(x):
    if isinstance(x, dict):
        return {k: to_json_safe(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [to_json_safe(v) for v in x]
    if isinstance(x, set | frozenset):
        return sorted(to_json_safe(v) for v in x)
    if hasattr(x, "__dict__"):
        return to_json_safe(vars(x))
    return x


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _v(x):
    return x.value if isinstance(x, Enum) else x


def _flatten_dict(d: dict, *, prefix: str = "") -> list[str]:
    parts = []

    for key in sorted(d):
        value = d[key]
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            parts.extend(_flatten_dict(value, prefix=full_key))
        elif isinstance(value, list):
            parts.append(f"{full_key}=[{','.join(map(str, value))}]")
        else:
            parts.append(f"{full_key}={value}")

    return parts


def _get_config_label(config) -> str:
    d = config.model_dump(mode="json", exclude_none=True)

    d.get("data", {}).pop("seed", None)

    algo_name = d.get("algo", {}).pop("name", "algo")
    setting = d.get("data", {}).pop("setting", "data")

    fields = _flatten_dict(d)

    return " | ".join([str(algo_name), str(setting), *fields])


def _format_mean_std(mean: float, std: float, n: int) -> str:
    if math.isnan(mean):
        return f"n/a (n={n})"
    if math.isnan(std):
        return f"{mean:.4f} ± n/a (n={n})"
    return f"{mean:.4f} ± {std:.4f} (n={n})"


def _metrics_to_float_dict(metrics_obj: Any) -> dict[str, float]:
    raw = dataclasses.asdict(metrics_obj)
    return {str(key): float(value) for key, value in raw.items()}


def _node_to_summary_var(node) -> str:
    if isinstance(node, tuple):
        return str(node[0])

    text = str(node)

    if ":" in text:
        return text.split(":", 1)[0]
    if "_lag" in text:
        return text.split("_lag", 1)[0]

    return text


def _compact_signatures(signatures: dict[Any, tuple[Any, ...]]) -> dict[Any, int]:
    label_by_signature: dict[tuple[Any, ...], int] = {}
    labels: dict[Any, int] = {}

    for key, signature in signatures.items():
        if signature not in label_by_signature:
            label_by_signature[signature] = len(label_by_signature)
        labels[key] = label_by_signature[signature]

    return labels


def _estimated_cmm_labels_by_target(est: CausalChange) -> dict[str, list[int]]:
    components = getattr(est, "mixture_components_", None)

    if components is None:
        return {}

    out: dict[str, list[int]] = {}

    for target, target_result in components.target_components.items():
        out[str(target)] = [int(label) for label in target_result.labels]

    return out


def _compute_cmm_mixture_metrics(
    *,
    true_labels_by_target: dict[str, list[int]],
    estimated_labels_by_target: dict[str, list[int]],
    targets: list[str] | None = None,
) -> dict[str, float]:
    if targets is None:
        targets = sorted(true_labels_by_target)

    metrics: dict[str, float] = {}
    ari_values: list[float] = []
    ami_values: list[float] = []
    nmi_values: list[float] = []

    for target in targets:
        if target not in true_labels_by_target:
            continue
        if target not in estimated_labels_by_target:
            continue

        true_labels = [int(x) for x in true_labels_by_target[target]]
        estimated_labels = [int(x) for x in estimated_labels_by_target[target]]

        if len(true_labels) != len(estimated_labels):
            metrics[f"cmm_mixture_{target}_valid"] = 0.0
            continue

        scores = compute_partition_metrics(true_labels, estimated_labels)

        metrics[f"cmm_mixture_{target}_ari"] = float(scores.ari)
        metrics[f"cmm_mixture_{target}_ami"] = float(scores.ami)
        metrics[f"cmm_mixture_{target}_nmi"] = float(scores.nmi)
        metrics[f"cmm_mixture_{target}_n_true_clusters"] = float(len(set(true_labels)))
        metrics[f"cmm_mixture_{target}_n_est_clusters"] = float(len(set(estimated_labels)))
        metrics[f"cmm_mixture_{target}_valid"] = 1.0

        ari_values.append(float(scores.ari))
        ami_values.append(float(scores.ami))
        nmi_values.append(float(scores.nmi))

    metrics["cmm_mixture_ari"] = float(np.nanmean(ari_values)) if ari_values else float("nan")
    metrics["cmm_mixture_ami"] = float(np.nanmean(ami_values)) if ami_values else float("nan")
    metrics["cmm_mixture_nmi"] = float(np.nanmean(nmi_values)) if nmi_values else float("nan")
    metrics["cmm_mixture_n_targets"] = float(len(ari_values))

    return metrics


def _estimated_context_labels_by_target(est: CausalChange) -> dict[str, dict[int, int]]:
    """Project cell clusters to one context label per target for benchmark metrics."""
    result = est.get_result()
    partitions = result.grid_clusters

    if partitions is None:
        return {}

    out: dict[str, dict[int, int]] = {}

    for target, mapping in partitions.cell_clusters.items():
        signatures: dict[Any, tuple[int, ...]] = {}

        for dataset_id, intervals in partitions.intervals_by_context.items():
            signatures[dataset_id] = tuple(
                int(mapping[GridCell(dataset_id=dataset_id, interval_id=interval_id)])
                for interval_id in range(len(intervals))
            )

        out[str(target)] = {
            int(dataset_id): int(label) for dataset_id, label in _compact_signatures(signatures).items()
        }

    return out


def _estimated_regime_labels_by_target(est: CausalChange) -> dict[str, dict[int, int]]:
    """Project cell clusters to one regime/interval label per target for benchmark metrics."""
    result = est.get_result()
    partitions = result.grid_clusters

    if partitions is None:
        return {}

    max_intervals = max(
        (len(intervals) for intervals in partitions.intervals_by_context.values()),
        default=0,
    )
    out: dict[str, dict[int, int]] = {}

    for target, mapping in partitions.cell_clusters.items():
        signatures: dict[int, tuple[tuple[str, int], ...]] = {}

        for interval_id in range(max_intervals):
            signatures[interval_id] = tuple(
                (
                    repr(dataset_id),
                    int(mapping[GridCell(dataset_id=dataset_id, interval_id=interval_id)]),
                )
                for dataset_id, intervals in partitions.intervals_by_context.items()
                if interval_id < len(intervals)
            )

        out[str(target)] = {
            int(interval_id): int(label) for interval_id, label in _compact_signatures(signatures).items()
        }

    return out


def _project_temporal_graph_to_summary(graph: nx.DiGraph) -> nx.DiGraph:
    summary = nx.DiGraph()

    for node in graph.nodes():
        summary.add_node(_node_to_summary_var(node))

    for u, v in graph.edges():
        uu = _node_to_summary_var(u)
        vv = _node_to_summary_var(v)

        # Summary DAG metrics usually ignore self-lag edges.
        if uu != vv:
            summary.add_edge(uu, vv)

    return summary
