from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterable
from itertools import product
from typing import Any, cast

import networkx as nx
from pydantic import BaseModel, ValidationError

from benchmarks.synthetic.generator_time import BenchmarkSample
from benchmarks.synthetic.generators import (
    sample_multi_continuous,
    sample_multi_temporal,
    sample_single_continuous,
    sample_single_temporal,
)
from benchmarks.synthetic.metrics import compute_metrics
from benchmarks.synthetic.metrics_time import (
    compute_changepoint_metrics,
    compute_target_partition_metrics,
    compute_target_regime_partition_metrics_over_time,
)
from benchmarks.utils import _pgmpy_graph_to_nx
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
from causalchange.config.cc_config import ChangepointMode
from causalchange.config.cc_types import ContextAggregation, GPType, ScoreType

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


def _estimated_context_labels_by_target(est: CausalChange) -> dict[str, dict[int, int]]:
    partitions = est.result.partitions
    return {
        str(target): {int(dataset_id): int(label) for dataset_id, label in labels.items()}
        for target, labels in partitions.contexts.items()
    }


def _estimated_regime_labels_by_target(est: CausalChange) -> dict[str, dict[int, int]]:
    partitions = est.result.partitions
    return {
        str(target): {int(regime_id): int(label) for regime_id, label in labels.items()}
        for target, labels in partitions.regimes.items()
    }


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
        return est_or_graph.graph
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


def run_algo(sample: BenchmarkSample, data_cfg: DataConfig, algo_cfg: AlgoConfig) -> CausalChange:
    from causalchange.config.cc_types import DataMode, GraphSearch

    data_mode = DataMode(data_cfg.setting)

    is_spacetime = algo_cfg.name == "spacetime"
    is_temporal = data_mode.is_temporal()

    spacetime_algo_cfg: SpaceTimeAlgoConfig | None = None
    if is_spacetime:
        spacetime_algo_cfg = cast(SpaceTimeAlgoConfig, algo_cfg)

    temporal_data_cfg: TemporalDataConfig | None = None
    if is_temporal:
        temporal_data_cfg = cast(TemporalDataConfig, data_cfg)

    graph_search = (
        GraphSearch.GLOBE
        if is_spacetime
        else (GraphSearch.TOPIC if algo_cfg.name in ("topic", "linc", "chain") else GraphSearch.SKIP)
    )

    if graph_search == GraphSearch.SKIP:
        raise ValueError(f"invalid: {algo_cfg.name}")

    aggregation = (
        ContextAggregation.LINC
        if algo_cfg.name == "linc"
        else (ContextAggregation.CHAIN if algo_cfg.name == "chain" else ContextAggregation.SKIP)
    )

    score_type = _resolve_score_type(algo_cfg.score_type)

    if is_temporal:
        assert temporal_data_cfg is not None
        assert spacetime_algo_cfg is not None
        tau_max = spacetime_algo_cfg.tau_max or temporal_data_cfg.tau_max
        d_min = temporal_data_cfg.min_segment_length
    else:
        tau_max = None
        d_min = 30

    if data_mode.is_context():
        context_data_cfg = cast(ContextDataConfig, data_cfg)
        context_col = context_data_cfg.context_col
    else:
        context_col = None

    if is_spacetime and is_temporal:
        assert spacetime_algo_cfg is not None
        assert sample.spacetime is not None

        if spacetime_algo_cfg.changepoint_mode == "oracle":
            changepoints = ChangepointMode.FIXED
            fixed_changepoints = sample.spacetime.changepoints
        elif spacetime_algo_cfg.changepoint_mode == "detect":
            changepoints = ChangepointMode.DETECT
            fixed_changepoints = None
        else:
            changepoints = ChangepointMode.NONE
            fixed_changepoints = None

        detect_contexts = spacetime_algo_cfg.detect_contexts
        detect_regimes = spacetime_algo_cfg.detect_regimes
    else:
        changepoints = ChangepointMode.NONE
        fixed_changepoints = None
        detect_contexts = False
        detect_regimes = False

    est = CausalChange(
        data_mode=data_mode,
        graph_search=graph_search,
        score_type=score_type,
        aggregation=aggregation,
        context_col=context_col,
        tau_max=tau_max,
        changepoints=changepoints,
        d_min=d_min,
        fixed_changepoints=fixed_changepoints,
        detect_contexts=detect_contexts,
        detect_regimes=detect_regimes,
    )

    return est.fit(sample.df)


def run_scoring(
    sample: BenchmarkSample,
    est: CausalChange,
    return_nx: bool = False,
) -> dict[str, float] | tuple[dict[str, float], nx.DiGraph]:
    est_nx = est.graph

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
                est.result.changepoints,
                tolerance=5,
            )
        )

        metrics.update(changepoint_metrics)
        spacetime_cfg = est.cfg.spacetime

        if spacetime_cfg.detect_contexts:
            context_partition_metrics = compute_target_partition_metrics(
                spacetime_sample.context_labels_by_target,
                _estimated_context_labels_by_target(est),
            )

            metrics["context_partition_ari"] = context_partition_metrics.ari_mean
            metrics["context_partition_ami"] = context_partition_metrics.ami_mean
            metrics["context_partition_nmi"] = context_partition_metrics.nmi_mean

        if spacetime_cfg.detect_regimes:
            regime_partition_metrics = compute_target_regime_partition_metrics_over_time(
                spacetime_sample.regime_labels_by_target,
                spacetime_sample.changepoints,
                _estimated_regime_labels_by_target(est),
                est.result.changepoints,
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
