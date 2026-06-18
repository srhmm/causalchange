from __future__ import annotations

import time
from collections.abc import Iterable
from itertools import product
from typing import Any, cast

import networkx as nx
from pydantic import ValidationError

from causalchange import CMM, Linc, SpaceTime, Topic
from causalchange.causal_change import CausalChange
from causalchange.config.benchmark_config import (
    AlgoConfig,
    BenchmarkConfig,
    CmmAlgoConfig,
    ContextDataConfig,
    DataConfig,
    MultiTemporalDataConfig,
    SpaceTimeAlgoConfig,
    TemporalDataConfig,
)
from causalchange.core.types import (
    MechanismClusteringScope,
    TabularContextMethod,
)
from experiments.benchmarks.synthetic.metrics import compute_metrics
from experiments.benchmarks.synthetic.metrics_time import (
    compute_changepoint_metrics,
    compute_target_partition_metrics,
    compute_target_regime_partition_metrics_over_time,
)
from experiments.benchmarks.synthetic.sample import BenchmarkSample
from experiments.benchmarks.synthetic.sample_tabular import (
    sample_mixed_continuous,
    sample_multi_continuous,
    sample_multi_temporal,
    sample_single_continuous,
    sample_single_temporal,
)
from experiments.benchmarks.utils import (
    _compute_cmm_mixture_metrics,
    _estimated_cmm_labels_by_target,
    _estimated_context_labels_by_target,
    _estimated_linc_context_labels_by_target,
    _estimated_regime_labels_by_target,
    _metrics_to_float_dict,
    _project_temporal_graph_to_summary,
)


def run_sampling(config: DataConfig) -> BenchmarkSample:
    sampler_by_setting = {
        "single": sample_single_continuous,
        "multi": sample_multi_continuous,
        "mixed": sample_mixed_continuous,
        "time": sample_single_temporal,
        "time-contexts": sample_multi_temporal,
    }

    try:
        sampling_fun = sampler_by_setting[config.setting]
    except KeyError as exc:
        raise NotImplementedError(f"Unknown sampling fun for {config.setting!r}") from exc

    result = sampling_fun(config)

    if config.setting in {"time", "time-contexts"}:
        return BenchmarkSample(
            df=result.df,
            true_summary_dag=result.true_summary_dag,
            spacetime=result,
            mixed=None,
            multi=None,
        )

    if config.setting == "mixed":
        return BenchmarkSample(
            df=result.df,
            true_summary_dag=result.true_summary_dag,
            spacetime=None,
            mixed=result,
            multi=None,
        )

    if config.setting == "multi":
        return BenchmarkSample(
            df=result.df,
            true_summary_dag=result.true_summary_dag,
            spacetime=None,
            mixed=None,
            multi=result,
        )

    df, true_g = result
    return BenchmarkSample(
        df=df,
        true_summary_dag=true_g,
        spacetime=None,
        mixed=None,
        multi=None,
    )


def run_algo(sample: BenchmarkSample, data_cfg: DataConfig, algo_cfg: AlgoConfig) -> CausalChange:
    if algo_cfg.name == "topic":
        return Topic(score_type=algo_cfg.score_type).fit(sample.df)

    if algo_cfg.name == "cmm":
        cmm_cfg = cast(CmmAlgoConfig, algo_cfg)
        return CMM(
            mix_type=cmm_cfg.mix_type,
            k_max=cmm_cfg.k_max,
            score_kwargs={
                "max_em_iter": cmm_cfg.max_em_iter,
                "n_init": cmm_cfg.n_init,
                "tol": cmm_cfg.tol,
                "ridge": cmm_cfg.ridge,
            },
        ).fit(sample.df)

    if algo_cfg.name == "linc":
        context_cfg = cast(ContextDataConfig, data_cfg)
        return Linc(
            score_type=algo_cfg.score_type,
            context_col=context_cfg.context_col,
        ).fit(sample.df)

    if algo_cfg.name != "spacetime":
        raise ValueError(f"invalid: {algo_cfg.name}")

    assert data_cfg.setting in {"time", "time-contexts"}
    spacetime_algo_cfg = cast(SpaceTimeAlgoConfig, algo_cfg)
    temporal_data_cfg = cast(TemporalDataConfig, data_cfg)

    kwargs = spacetime_algo_cfg.model_dump(
        exclude={"name"},
        exclude_none=True,
    )

    kwargs["data_mode"] = data_cfg.setting
    kwargs["tau_max"] = spacetime_algo_cfg.tau_max or temporal_data_cfg.tau_max
    kwargs["d_min"] = spacetime_algo_cfg.d_min or temporal_data_cfg.min_segment_length

    if data_cfg.setting == "time-contexts":
        kwargs["context_col"] = cast(MultiTemporalDataConfig, data_cfg).context_col
    else:
        kwargs["context_col"] = "context"

    if kwargs.get("changepoint_mode") == "fixed" and kwargs.get("fixed_changepoints") is None:
        if sample.spacetime is None:
            raise ValueError("Cannot inject fixed changepoints because sample.spacetime is None.")
        kwargs["fixed_changepoints"] = sample.spacetime.changepoints

    return SpaceTime(**kwargs).fit(sample.df)


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

    if sample.mixed is not None:
        cmm_metrics = _compute_cmm_mixture_metrics(
            true_labels_by_target=sample.mixed.labels_by_target,
            estimated_labels_by_target=_estimated_cmm_labels_by_target(est),
            targets=sample.mixed.mixed_targets,
        )
        metrics.update(cmm_metrics)

    if sample.multi is not None and getattr(est.cfg, "context_combination_method", None) == TabularContextMethod.LINC:
        linc_context_partition_metrics = compute_target_partition_metrics(
            sample.multi.context_labels_by_target,
            _estimated_linc_context_labels_by_target(est),
        )

        metrics["context_partition_ari"] = linc_context_partition_metrics.ari_mean
        metrics["context_partition_ami"] = linc_context_partition_metrics.ami_mean
        metrics["context_partition_nmi"] = linc_context_partition_metrics.nmi_mean

        for target, value in linc_context_partition_metrics.ari_by_target.items():
            metrics[f"context_partition_{target}_ari"] = float(value)
        for target, value in linc_context_partition_metrics.ami_by_target.items():
            metrics[f"context_partition_{target}_ami"] = float(value)
        for target, value in linc_context_partition_metrics.nmi_by_target.items():
            metrics[f"context_partition_{target}_nmi"] = float(value)

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


def _product_dict(d: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    for combo in product(*vals):
        yield dict(zip(keys, combo, strict=False))


def iter_valid_configs(grid: dict[str, Any]):
    data_grid = grid.get("data", {})
    algo_grid = grid.get("algo", {})

    for data_opt in _product_dict(data_grid):
        for algo_opt in _product_dict(algo_grid):
            candidate = {
                "data": dict(data_opt),
                "algo": dict(algo_opt),
            }

            try:
                yield BenchmarkConfig.model_validate(candidate)
            except ValidationError as exc:
                print("Invalid benchmark candidate:")
                print(candidate)
                print(exc)
                continue
