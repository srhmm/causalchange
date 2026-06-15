import math
from dataclasses import dataclass
from typing import Any

import networkx as nx


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

def _short_config_label(config: dict[str, Any]) -> str:
    data = config["data"]
    algo = config["algo"]

    parts = [
        str(algo.get("name")),
        str(data.get("setting")),
        f"fun={data.get('nonlinearity')}",
        f"score={algo.get('score_type')}",
        f"nn={data.get('n_nodes')}",
        f"p={data.get('edge_prob')}",
    ]

    if "n_samples" in data:
        parts.append(f"n={data.get('n_samples')}")

    if "n_samples_per_context" in data:
        parts.append(f"nctx={data.get('n_samples_per_context')}")

    if "n_contexts" in data:
        parts.append(f"ctx={data.get('n_contexts')}")

    if "tau_max" in data:
        parts.append(f"tau={data.get('tau_max')}")

    if "n_changepoints" in data:
        parts.append(f"cp={data.get('n_changepoints')}")

    return " | ".join(parts)

def _format_mean_std(mean: float, std: float, n: int) -> str:
    if math.isnan(mean):
        return f"n/a (n={n})"
    if math.isnan(std):
        return f"{mean:.4f} ± n/a (n={n})"
    return f"{mean:.4f} ± {std:.4f} (n={n})"
