from __future__ import annotations

import time
from itertools import product
from typing import Any, Iterable

import networkx as nx
import pandas as pd
from pydantic import BaseModel, ValidationError

from benchmarks.synthetic.generators import (
    sample_linear_gaussian,
    sample_nonlinear_additive,
    sample_multicontext_linear_gaussian_interventional,
    sample_multicontext_nonlinear_additive_interventional,
)

from  benchmarks.synthetic.metrics import compute_metrics

from benchmarks.benchmark_configs import BenchmarkConfig, DataConfig, ScoringConfig, AlgoConfig, \
    LincAlgoConfig, TopicAlgoConfig, PcAlgoConfig, MultiNonlinearDataConfig, MultiLinearDataConfig, \
    SingleNonlinearDataConfig, SingleLinearDataConfig

from causalchange._cc_types import MixingType
from causalchange.causal_change import CausalChange


def run_sampling(config: DataConfig):
    if isinstance(config, SingleLinearDataConfig):
        return sample_linear_gaussian(
            n_samples=config.n_samples,
            n_nodes=config.n_nodes,
            edge_prob=config.edge_prob,
            seed=config.seed,
            weight_scale=config.weight_scale,
            noise_scale=config.noise_scale,
        )

    if isinstance(config, SingleNonlinearDataConfig):
        return sample_nonlinear_additive(
            n_samples=config.n_samples,
            n_nodes=config.n_nodes,
            edge_prob=config.edge_prob,
            seed=config.seed,
            weight_scale=config.weight_scale,
            noise_scale=config.noise_scale,
            nonlinearity=config.nonlinearity,
        )

    if isinstance(config, MultiLinearDataConfig):
        df, _, true_target, _, _ = sample_multicontext_linear_gaussian_interventional(
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
        return df, true_target

    if isinstance(config, MultiNonlinearDataConfig):
        assert config.alt_nonlinearity is not None
        df, _, true_target, _, _ = sample_multicontext_nonlinear_additive_interventional(
            n_samples_per_context=config.n_samples_per_context,
            n_nodes=config.n_nodes,
            edge_prob=config.edge_prob,
            n_contexts=config.n_contexts,
            seed=config.seed,
            context_col=config.context_col,
            n_intervened_per_context=config.n_intervened_per_context,
            weight_scale=config.weight_scale,
            noise_scale=config.noise_scale,
            nonlinearity=config.nonlinearity,
            intervention_type=config.intervention_type,
            weight_scale_intervened=config.weight_scale_intervened,
            alt_nonlinearity=config.alt_nonlinearity,
            shift_scale=config.shift_scale,
            noise_scale_intervened=config.noise_scale_intervened,
        )
        return df, true_target

    raise RuntimeError(f"Unknown data config: {config!r}")



def _map_score_type(name: str):
    """
    Maps benchmark grid score_type strings to causalchange ScoreType/GPType values.
    """
    from causalchange._cc_types import ScoreType, GPType
    s = str(name).lower()
    if s == "lin":
        return ScoreType.LIN
    if s == "gam":
        return ScoreType.GAM
    if s == "spline":
        return ScoreType.SPLINE
    if s == "krr":
        return ScoreType.KRR
    if s == "gp":
        return GPType.EXACT
    if s in ("ff", "rff"):
        return GPType.FOURIER
    raise ValueError(f"Unknown score_type: {name!r}")


def run_algo(df: pd.DataFrame, data_cfg: DataConfig, algo_cfg: AlgoConfig) -> Any:
    # Determine DataMode from setting string.
    from causalchange._cc_types import DataMode, GraphSearch

    if data_cfg.setting == "single":
        data_mode = DataMode.IID
    elif data_cfg.setting == "multi":
        data_mode = DataMode.CONTEXTS
    elif data_cfg.setting == "time":
        data_mode = DataMode.TIME
    elif data_cfg.setting == "time_contexts":
        data_mode = DataMode.TIME_CONTEXTS
    else:
        raise ValueError(f"Unknown data setting: {data_cfg.setting!r}")

    # Graph search: currently topic/globe only
    graph_search = GraphSearch.TOPIC if algo_cfg.name in ("topic", "linc", "spacetime", "spacetime_c") else GraphSearch.TOPIC

    # score_type from algo config
    score_type = _map_score_type(getattr(algo_cfg, "score_type", "gam"))

    context_col = getattr(data_cfg, "context_col", "context")
    tau_max = getattr(data_cfg, "tau_max", 1)

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

def _pgmpy_graph_to_nx(dag: Any) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from([str(n) for n in dag.nodes()])
    g.add_edges_from([(str(u), str(v)) for (u, v) in dag.edges()])
    return g

def run_scoring(true_g, est_dag, scoring_cfg: ScoringConfig) -> dict[str, float]:
    est_nx = _pgmpy_graph_to_nx(est_dag)
    m = compute_metrics(true_g, est_nx)

    out: dict[str, float] = {}
    if "shd" in scoring_cfg.metrics:
        out["shd"] = float(m.shd)
    if "edge_f1" in scoring_cfg.metrics:
        out["edge_f1"] = float(m.edge_f1)
    if "skel_f1" in scoring_cfg.metrics:
        out["skel_f1"] = float(m.skel_f1)

    return out


def run_on_config(cfg: BenchmarkConfig) -> dict[str, float]:
    df, true_g = run_sampling(cfg.data)

    t0 = time.perf_counter()
    est_dag = run_algo(df, cfg.data, cfg.algo)
    t1 = time.perf_counter()

    metrics = run_scoring(true_g, est_dag, cfg.scoring)
    if "time_s" in cfg.scoring.metrics:
        metrics["time_s"] = float(t1 - t0)

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
        m = _data_model_for(data_opt0)
        if m is None:
            continue
        data_opt = _filter_to_model_fields(m, data_opt0)

        for algo_opt0 in _product_dict(algo_grid):
            algo_opt0 = dict(algo_opt0)  # safety copy
            name = algo_opt0.get("name")

            if name == "linc":
                algo_opt = _filter_to_model_fields(LincAlgoConfig, algo_opt0)
            elif name == "topic":
                algo_opt = _filter_to_model_fields(TopicAlgoConfig, algo_opt0)
            elif name == "pc":
                algo_opt = _filter_to_model_fields(PcAlgoConfig, algo_opt0)
            else:
                continue

            for scoring_opt0 in scoring_options:
                scoring_opt = dict(scoring_opt0)
                candidate = {"data": data_opt, "algo": algo_opt, "scoring": scoring_opt}

                try:
                    yield BenchmarkConfig.model_validate(candidate)
                except ValidationError:
                    continue



def _data_model_for(data_opt: dict[str, Any]) -> type[BaseModel] | None:
    setting = data_opt.get("setting")
    linearity = data_opt.get("linearity")

    if setting == "single" and linearity == "linear":
        return SingleLinearDataConfig
    if setting == "single" and linearity == "nonlinear":
        return SingleNonlinearDataConfig
    if setting == "multi" and linearity == "linear":
        return MultiLinearDataConfig
    if setting == "multi" and linearity == "nonlinear":
        return MultiNonlinearDataConfig

    return None