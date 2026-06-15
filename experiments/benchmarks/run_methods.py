from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterable
from itertools import product
from typing import Any, cast

import networkx as nx
from pydantic import BaseModel, ValidationError

from causalchange import Linc, SpaceTime, Topic
from causalchange.causal_change import CausalChange
from causalchange.config.benchmark_config import (
    AlgoConfig,
    BenchmarkConfig,
    ChainAlgoConfig,
    DataConfig,
    LincAlgoConfig,
    MixedDataConfig,
    MultiDataConfig,
    MultiTemporalDataConfig,
    SingleDataConfig,
    SingleTemporalDataConfig,
    SpaceTimeAlgoConfig,
    TopicAlgoConfig,
)
from causalchange.config.causal_change_config import CausalChangeConfigTabular
from causalchange.core.results import GridCell
from causalchange.core.types import (
    DataMode,
    GPType,
    GraphSearch,
    MechanismClusteringScope,
    ScoreType,
    TabularContextMethod,
    TabularContextMode,
)
from experiments.benchmarks.synthetic.generator_time import BenchmarkSample
from experiments.benchmarks.synthetic.generators import (
    sample_multi_continuous,
    sample_multi_temporal,
    sample_single_continuous,
    sample_single_temporal,
)
from experiments.benchmarks.synthetic.metrics import compute_metrics
from experiments.benchmarks.synthetic.metrics_time import (
    compute_changepoint_metrics,
    compute_target_partition_metrics,
    compute_target_regime_partition_metrics_over_time,
)
from experiments.benchmarks.utils import _pgmpy_graph_to_nx

TemporalDataConfig = SingleTemporalDataConfig | MultiTemporalDataConfig
ContextDataConfig = MultiDataConfig | MultiTemporalDataConfig | MixedDataConfig


def _metrics_to_float_dict(metrics_obj: Any) -> dict[str, float]:
    raw = dataclasses.asdict(metrics_obj)
    return {str(key): float(value) for key, value in raw.items()}


def _node_to_summary_var(node) -> str:
    if isinstance(node, tuple):
        return str(node[0])

    text = str(node)

    # support for possible textual lag formats.
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


def _estimator_to_nx(est_or_graph: Any) -> nx.DiGraph:
    if isinstance(est_or_graph, CausalChange):
        return est_or_graph.graph_
    if isinstance(est_or_graph, nx.DiGraph):
        return est_or_graph
    return _pgmpy_graph_to_nx(est_or_graph)


def _resolve_score_type(value: str) -> ScoreType | GPType:
    if value in (GPType.EXACT.value, GPType.FOURIER.value):
        return GPType(value)
    return ScoreType(value)


def run_sampling(config: DataConfig) -> BenchmarkSample:
    sampling_fun = (
        sample_single_continuous
        if config.setting == "single"
        else (
            sample_multi_continuous
            if config.setting == "multi"
            else (
                sample_single_temporal
                if config.setting == "time"
                else (sample_multi_temporal if config.setting == "time-contexts" else None)
            )
        )
    )

    if sampling_fun is None:
        raise NotImplementedError(f"Unknown sampling fun for {config.setting!r}")

    result = sampling_fun(config)

    if config.setting in {"time", "time-contexts"}:
        return BenchmarkSample(
            df=result.df,
            true_summary_dag=result.true_summary_dag,
            spacetime=result,
        )

    df, true_g = result
    return BenchmarkSample(
        df=df,
        true_summary_dag=true_g,
        spacetime=None,
    )


def _data_mode_from_setting(setting: str) -> DataMode:
    mapping = {
        "single": DataMode.TABULAR,
        "multi": DataMode.TAB_CONTEXTS,
        "mixed": DataMode.TABULAR,
        "time": DataMode.TIME,
        "time-contexts": DataMode.TIME_CONTEXTS,
    }
    try:
        return mapping[setting]
    except KeyError as exc:
        raise ValueError(f"Unknown data setting: {setting!r}") from exc


def _clustering_scope_from_flags(*, contexts: bool, regimes: bool) -> MechanismClusteringScope:
    if contexts and regimes:
        return MechanismClusteringScope.REGIMES_CONTEXTS
    if contexts:
        return MechanismClusteringScope.CONTEXTS
    if regimes:
        return MechanismClusteringScope.REGIMES
    return MechanismClusteringScope.SKIP


def _changepoint_method_for_mode(mode: str) -> str:
    return "pelt" if mode == "detect" else "skip"


def _changepoint_scope_for_mode(mode: str, data_mode: str) -> str:
    if mode == "skip":
        return "skip"

    # Default benchmark behavior: detect/fix a shared global set of changepoints.
    return "global"


def _public_data_mode(data_mode: DataMode) -> str:
    if data_mode == DataMode.TIME:
        return "time"
    if data_mode == DataMode.TIME_CONTEXTS:
        return "time-contexts"
    raise ValueError(f"Expected temporal data mode, got {data_mode!r}.")


def _public_changepoint_mode(value: str) -> str:
    if value == "oracle":
        return "fixed"
    if value == "none":
        return "skip"
    if value == "detect":
        return "detect"
    raise ValueError(f"Unknown changepoint mode: {value!r}")


def _public_clustering_scope(*, contexts: bool, regimes: bool) -> str:
    if contexts and regimes:
        return "regimes-contexts"
    if contexts:
        return "contexts"
    if regimes:
        return "regimes"
    return "skip"


def run_algo(sample: BenchmarkSample, data_cfg: DataConfig, algo_cfg: AlgoConfig) -> CausalChange:
    data_mode = _data_mode_from_setting(data_cfg.setting)
    score_type = algo_cfg.score_type

    if algo_cfg.name == "topic":
        return Topic(score_type=score_type).fit(sample.df)

    if algo_cfg.name == "linc":
        context_cfg = cast(ContextDataConfig, data_cfg)
        return Linc(score_type=score_type, context_col=context_cfg.context_col).fit(sample.df)

    if algo_cfg.name == "chain":
        context_cfg = cast(ContextDataConfig, data_cfg)
        cfg = CausalChangeConfigTabular(
            data_mode=DataMode.TAB_CONTEXTS,
            graph_search=GraphSearch.TOPIC,
            score_type=_resolve_score_type(score_type),
            context_mode=TabularContextMode.ORACLE,
            context_combination_method=TabularContextMethod.CHAIN,
            context_col=context_cfg.context_col,
        )
        return CausalChange(cfg).fit(sample.df)

    if algo_cfg.name != "spacetime":
        raise ValueError(f"invalid: {algo_cfg.name}")

    if not data_mode.is_temporal():
        raise ValueError("SpaceTime benchmarks require temporal data.")

    spacetime_algo_cfg = cast(SpaceTimeAlgoConfig, algo_cfg)
    temporal_data_cfg = cast(TemporalDataConfig, data_cfg)

    tau_max = spacetime_algo_cfg.tau_max or temporal_data_cfg.tau_max
    changepoint_mode = _public_changepoint_mode(spacetime_algo_cfg.changepoint_mode)
    fixed_changepoints = (
        sample.spacetime.changepoints if changepoint_mode == "fixed" and sample.spacetime is not None else None
    )
    regimes = bool(getattr(temporal_data_cfg, "n_changepoints", 0))
    contexts = (
            data_mode.is_context()
            and getattr(temporal_data_cfg, "n_context_clusters", 1) > 1
    )

    clustering_scope = _public_clustering_scope(
        contexts=contexts,
        regimes=regimes,
    )
    clustering_method = "mechanism-clustering" if clustering_scope != "skip" else "skip"

    return SpaceTime(
        data_mode=_public_data_mode(data_mode),
        score_type=score_type,
        tau_max=tau_max,
        context_col=(cast(ContextDataConfig, data_cfg).context_col if data_mode.is_context() else "context"),
        changepoint_mode=changepoint_mode,
        changepoint_scope=("global" if changepoint_mode != "skip" else "skip"),
        changepoint_method=("pelt" if changepoint_mode == "detect" else "skip"),
        fixed_changepoints=fixed_changepoints,
        clustering_scope=clustering_scope,
        clustering_method=clustering_method,
        testing_method="skip",
        d_min=temporal_data_cfg.min_segment_length,
    ).fit(sample.df)


def run_scoring(
    sample: BenchmarkSample,
    est: CausalChange,
    return_nx: bool = False,
) -> dict[str, float] | tuple[dict[str, float], nx.DiGraph]:
    est_nx = est.graph_

    if any(isinstance(node, tuple) for node in est_nx.nodes()):
        est_summary = _project_temporal_graph_to_summary(est_nx)
    else:
        est_summary = est_nx

    summary_metrics = _metrics_to_float_dict(compute_metrics(sample.true_summary_dag, est_summary))

    metrics: dict[str, float] = {}
    metrics.update(summary_metrics)

    for key, value in summary_metrics.items():
        metrics[f"summary_{key}"] = value

    spacetime_sample = sample.spacetime
    if spacetime_sample is not None:
        wcg_metrics = _metrics_to_float_dict(compute_metrics(spacetime_sample.true_wcg, est_nx))

        for key, value in wcg_metrics.items():
            metrics[f"wcg_{key}"] = value

        changepoint_metrics = _metrics_to_float_dict(
            compute_changepoint_metrics(
                spacetime_sample.changepoints,
                est.changepoints_,
                tolerance=5,
            )
        )

        metrics.update(changepoint_metrics)
        spacetime_cfg = est.cfg
        clustering_scope = getattr(spacetime_cfg, "clustering_scope", MechanismClusteringScope.SKIP)

        if clustering_scope.detects_contexts():
            context_partition_metrics = compute_target_partition_metrics(
                spacetime_sample.context_labels_by_target,
                _estimated_context_labels_by_target(est),
            )

            metrics["context_partition_ari"] = context_partition_metrics.ari_mean
            metrics["context_partition_ami"] = context_partition_metrics.ami_mean
            metrics["context_partition_nmi"] = context_partition_metrics.nmi_mean

        if clustering_scope.detects_regimes():
            regime_partition_metrics = compute_target_regime_partition_metrics_over_time(
                spacetime_sample.regime_labels_by_target,
                spacetime_sample.changepoints,
                _estimated_regime_labels_by_target(est),
                est.changepoints_,
                n_samples=len(spacetime_sample.time_regime_labels),
            )

            metrics["regime_partition_ari"] = regime_partition_metrics.ari_mean
            metrics["regime_partition_ami"] = regime_partition_metrics.ami_mean
            metrics["regime_partition_nmi"] = regime_partition_metrics.nmi_mean

    if return_nx:
        return metrics, est_nx

    return metrics


def run_on_config(
    cfg: BenchmarkConfig,
    return_nx=False,
) -> dict[str, float] | tuple[dict[str, float], nx.DiGraph]:
    sample = run_sampling(cfg.data)

    t0 = time.perf_counter()
    est_dag = run_algo(sample, cfg.data, cfg.algo)
    t1 = time.perf_counter()

    metrics, est_nx = run_scoring(sample, est_dag, True)
    metrics["time_s"] = float(t1 - t0)

    if return_nx:
        return metrics, est_nx

    return metrics


def _filter_to_model_fields(model_cls: type[BaseModel], data: dict[str, Any]) -> dict[str, Any]:
    allowed = set(model_cls.model_fields.keys())
    return {k: v for k, v in data.items() if k in allowed}


def _product_dict(d: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    for combo in product(*vals):
        yield dict(zip(keys, combo, strict=False))


def iter_valid_configs(grid: dict[str, Any]):
    data_grid = grid.get("data", {})
    algo_grid = grid.get("algo", {})

    for data_opt0 in _product_dict(data_grid):
        data_opt0 = dict(data_opt0)
        setting = data_opt0.get("setting")
        # funform = data_opt0.get("nonlinearity")
        model = (
            SingleDataConfig
            if setting == "single"
            else (
                MultiDataConfig
                if setting == "multi"
                else (
                    SingleTemporalDataConfig
                    if setting == "time"
                    else (
                        MultiTemporalDataConfig
                        if setting == "time-contexts"
                        else MixedDataConfig
                        if setting == "mixed"
                        else None
                    )
                )
            )
        )
        if model is None:
            raise ValueError(setting)

        data_opt = _filter_to_model_fields(model, data_opt0)

        for algo in _product_dict(algo_grid):
            algo = dict(algo)  # ?
            name = algo.get("name")

            algo_parent = (
                _filter_to_model_fields(LincAlgoConfig, algo)
                if name == "linc"
                else (
                    _filter_to_model_fields(ChainAlgoConfig, algo)
                    if name == "chain"
                    else (
                        _filter_to_model_fields(TopicAlgoConfig, algo)
                        if name == "topic"
                        else (_filter_to_model_fields(SpaceTimeAlgoConfig, algo) if name == "spacetime" else None)
                    )
                )
            )
            if algo_parent is None:
                raise ValueError(algo)
            candidate = {
                "data": data_opt,
                "algo": algo_parent,
            }

            try:
                yield BenchmarkConfig.model_validate(candidate)
            except ValidationError as exc:
                print("Invalid benchmark candidate:")
                print(candidate)
                print(exc)
                continue
