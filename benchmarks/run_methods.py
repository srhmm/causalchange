from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Iterable
from itertools import product
from typing import Any

import networkx as nx
import pandas as pd
from pydantic import BaseModel, ValidationError

from benchmarks.synthetic.generators import (
    sample_multi_continuous,
    sample_multi_temporal,
    sample_single_continuous,
    sample_single_temporal,
)
from benchmarks.synthetic.metrics import compute_metrics
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
    ScoringConfig,
    SingleDataConfig,
    SingleTemporalDataConfig,
    SpaceTimeAlgoConfig,
    SpaceTimeCAlgoConfig,
    TopicAlgoConfig,
)
from causalchange.config.cc_types import ContextAggregation, GPType, ScoreType


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


def run_sampling(config: DataConfig):
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

    df, true_g = sampling_fun(config)
    return df, true_g


def run_algo(df: pd.DataFrame, data_cfg: DataConfig, algo_cfg: AlgoConfig) -> Any:
    from causalchange.config.cc_types import DataMode, GraphSearch

    data_mode = DataMode(data_cfg.setting)
    graph_search = (
        GraphSearch.GLOBE
        if algo_cfg.name in ("spacetime", "spacetime-c")
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
    tau_max = getattr(algo_cfg, "tau_max", None) or getattr(data_cfg, "tau_max", None)
    context_col = getattr(data_cfg, "context_col", None)

    logging.basicConfig(level=logging.DEBUG)
    vb = 2
    est = CausalChange(
        data_mode=data_mode,
        graph_search=graph_search,
        score_type=score_type,
        aggregation=aggregation,
        context_col=context_col,
        tau_max=tau_max,
        vb=vb,
    )
    return est.fit(df)


def run_scoring(
    true_g,
    est_dag,
    scoring_cfg: ScoringConfig,
    return_nx=False,
) -> dict[str, float] | tuple[dict[str, float], nx.DiGraph]:
    est_nx = _estimator_to_nx(est_dag)
    graph_metrics = compute_metrics(true_g, est_nx)

    metrics = dataclasses.asdict(graph_metrics)
    ret_metrics = {ky: val for ky, val in metrics.items() if ky in scoring_cfg.metrics}

    if return_nx:
        return ret_metrics, est_nx
    return ret_metrics


def run_on_config(
    cfg: BenchmarkConfig,
    return_nx=False,
) -> dict[str, float] | tuple[dict[str, float], nx.DiGraph]:
    df, true_g = run_sampling(cfg.data)

    t0 = time.perf_counter()
    est_dag = run_algo(df, cfg.data, cfg.algo)
    t1 = time.perf_counter()

    metrics, est_nx = run_scoring(true_g, est_dag, cfg.scoring, True)
    if "time_s" in cfg.scoring.metrics:
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
    scoring_grid = grid.get("scoring", {})

    scoring_options = list(_product_dict(scoring_grid)) if scoring_grid else [{}]

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
                        else (
                            _filter_to_model_fields(SpaceTimeAlgoConfig, algo)
                            if name == "spacetime"
                            else (
                                _filter_to_model_fields(SpaceTimeCAlgoConfig, algo) if name == "spacetime-c" else None
                            )
                        )
                    )
                )
            )
            if algo_parent is None:
                raise ValueError(algo)

            for scoring_opt0 in scoring_options:
                scoring_opt = dict(scoring_opt0)
                candidate = {
                    "data": data_opt,
                    "algo": algo_parent,
                    "scoring": scoring_opt,
                }

                try:
                    yield BenchmarkConfig.model_validate(candidate)
                except ValidationError:
                    continue
