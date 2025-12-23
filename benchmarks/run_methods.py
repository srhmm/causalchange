from __future__ import annotations

import time
from itertools import product
from typing import Any, Iterable

import networkx as nx
import pandas as pd
from pydantic import BaseModel, ValidationError

from benchmarks.synthetic.generators import (
    sample_single_continuous,
    sample_multi_continuous, sample_single_temporal, sample_multi_temporal,
)

from  benchmarks.synthetic.metrics import compute_metrics

from benchmarks.benchmark_configs import BenchmarkConfig, DataConfig, ScoringConfig, AlgoConfig, \
    LincAlgoConfig, TopicAlgoConfig, \
    SingleDataConfig, MultiDataConfig, MultiTemporalDataConfig, SingleTemporalDataConfig, SpaceTimeAlgoConfig, \
    SpaceTimeCAlgoConfig, MixedDataConfig
from benchmarks.utils import _pgmpy_graph_to_nx

from causalchange._cc_types import MixingType, ScoreType
from causalchange.causal_change import CausalChange


def run_sampling(config: DataConfig):
    sampling_fun = sample_single_continuous if isinstance(config, SingleDataConfig) \
        else sample_multi_continuous if isinstance(config, MultiDataConfig) \
        else sample_single_temporal if isinstance(config, SingleTemporalDataConfig) \
        else sample_multi_temporal if isinstance(config, MultiTemporalDataConfig)  else None
    if sampling_fun is None:  raise RuntimeError(f"Unknown data config: {config!r}")

    df, true_g = sampling_fun(config)
    return df, true_g


def run_algo(df: pd.DataFrame, data_cfg: DataConfig, algo_cfg: AlgoConfig) -> Any:
    from causalchange._cc_types import DataMode, GraphSearch

    data_mode = DataMode(data_cfg.setting)
    graph_search = GraphSearch.TOPIC if algo_cfg.name in ("topic", "linc", "spacetime", "spacetime-c") else GraphSearch.TOPIC

    score_type = ScoreType (getattr(algo_cfg, "score_type", "gam"))
    tau_max = getattr(algo_cfg, "tau_max", 2)
    context_col = getattr(data_cfg, "context_col", "context")

    est = CausalChange(
        data_mode=data_mode,
        graph_search=graph_search,
        score_type=score_type,
        mixing_type=MixingType.SKIP,
        context_col=context_col,
        tau_max=tau_max,
        vb=0,
    )
    return est.fit(df)



def run_scoring(true_g, est_dag, scoring_cfg: ScoringConfig, return_nx=False) -> dict[str, float] | [dict[str, float], nx.DiGraph]:
    est_nx = _pgmpy_graph_to_nx(est_dag)
    graph_metrics = compute_metrics(true_g, est_nx)

    metrics: dict[str, float] = {}
    if "shd" in scoring_cfg.metrics: metrics["shd"] = float(graph_metrics.shd)
    if "edge_f1" in scoring_cfg.metrics: metrics["edge_f1"] = float(graph_metrics.edge_f1)
    if "skel_f1" in scoring_cfg.metrics: metrics["skel_f1"] = float(graph_metrics.skel_f1)

    if return_nx : return metrics, est_nx
    return metrics


def run_on_config(cfg: BenchmarkConfig, return_nx=False) -> dict[str, float]| [dict[str, float], nx.DiGraph]:
    df, true_g = run_sampling(cfg.data)

    t0 = time.perf_counter()
    est_dag = run_algo(df, cfg.data, cfg.algo)
    t1 = time.perf_counter()

    metrics, est_nx = run_scoring(true_g, est_dag, cfg.scoring, True)
    if "time_s" in cfg.scoring.metrics:
        metrics["time_s"] = float(t1 - t0)

    if return_nx : return metrics, est_nx
    return metrics



def _filter_to_model_fields(model_cls: type[BaseModel], data: dict[str, Any]) -> dict[str, Any]:
    allowed = set(model_cls.model_fields.keys())
    return {k: v for k, v in data.items() if k in allowed}


def _product_dict(d: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    for combo in product(*vals):
        yield dict(zip(keys, combo))


def iter_valid_configs(grid: dict[str, Any]):
    data_grid = grid.get("data", {})
    algo_grid = grid.get("algo", {})
    scoring_grid = grid.get("scoring", {})

    scoring_options = list(_product_dict(scoring_grid)) if scoring_grid else [{}]

    for data_opt0 in _product_dict(data_grid):
        data_opt0 = dict(data_opt0)
        setting = data_opt0.get("setting")
        model = (
            SingleDataConfig if setting == "single"
            else MultiDataConfig if setting == "multi"
            else SingleTemporalDataConfig if setting == "time"
            else MultiTemporalDataConfig if setting == "time-contexts"
            else MixedDataConfig if setting == "mixed"
            else None
        )
        if model is None: raise ValueError(setting)

        data_opt = _filter_to_model_fields(model, data_opt0)

        for algo in _product_dict(algo_grid):
            algo = dict(algo)  #?
            name = algo.get("name")

            algo_parent = _filter_to_model_fields(LincAlgoConfig, algo)if name == "linc" else \
                _filter_to_model_fields(TopicAlgoConfig, algo) if name == "topic" else \
                    _filter_to_model_fields(SpaceTimeAlgoConfig, algo)if name == "spacetime" else \
                        _filter_to_model_fields(SpaceTimeCAlgoConfig, algo) if name == "spacetime-c"  else None
            if algo_parent is None: raise ValueError(algo)

            for scoring_opt0 in scoring_options:
                scoring_opt = dict(scoring_opt0)
                candidate = {"data": data_opt, "algo": algo_parent, "scoring": scoring_opt}

                try:
                    yield BenchmarkConfig.model_validate(candidate)
                except ValidationError:
                    continue
