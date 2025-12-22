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

from benchmarks.synthetic.metrics import compute_metrics

from benchmarks.benchmark_configs import (
    AlgoConfig,
    BenchmarkConfig,
    DataConfig,
    LincAlgoConfig,
    MultiLinearDataConfig,
    MultiNonlinearDataConfig,
    MultiTemporalLinearDataConfig,
    MultiTemporalNonlinearDataConfig,
    ScoringConfig,
    SingleLinearDataConfig,
    SingleNonlinearDataConfig,
    SingleTemporalLinearDataConfig,
    SingleTemporalNonlinearDataConfig,
    SpaceTimeAlgoConfig,
    SpaceTimeCAlgoConfig,
    TopicAlgoConfig,
)

from causalchange.causal_change import CausalChange
from causalchange._cc_types import DataMode, GraphSearch

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

    # ---- Time series stubs (data generation to be implemented later) ----
    # For now, we reuse the IID / multi-context generators and interpret the rows as time steps.

    if isinstance(config, SingleTemporalLinearDataConfig):
        return sample_linear_gaussian(
            n_samples=config.n_samples,
            n_nodes=config.n_nodes,
            edge_prob=config.edge_prob,
            seed=config.seed,
            weight_scale=config.weight_scale,
            noise_scale=config.noise_scale,
        )

    if isinstance(config, SingleTemporalNonlinearDataConfig):
        return sample_nonlinear_additive(
            n_samples=config.n_samples,
            n_nodes=config.n_nodes,
            edge_prob=config.edge_prob,
            seed=config.seed,
            weight_scale=config.weight_scale,
            noise_scale=config.noise_scale,
            nonlinearity=config.nonlinearity,
        )

    if isinstance(config, MultiTemporalLinearDataConfig):
        df, _, true_target, _, _ = sample_multicontext_linear_gaussian_interventional(
            n_samples_per_context=config.n_samples_per_context,
            n_nodes=config.n_nodes,
            edge_prob=config.edge_prob,
            n_contexts=config.n_contexts,
            seed=config.seed,
            context_col=config.context_col,
            n_intervened_per_context=1,
            weight_scale=config.weight_scale,
            noise_scale=config.noise_scale,
            intervention_type="soft_weight",
            weight_scale_intervened=config.weight_scale,
            shift_scale=2.0,
            noise_scale_intervened=None,
        )
        return df, true_target

    if isinstance(config, MultiTemporalNonlinearDataConfig):
        df, _, true_target, _, _ = sample_multicontext_nonlinear_additive_interventional(
            n_samples_per_context=config.n_samples_per_context,
            n_nodes=config.n_nodes,
            edge_prob=config.edge_prob,
            n_contexts=config.n_contexts,
            seed=config.seed,
            context_col=config.context_col,
            n_intervened_per_context=1,
            weight_scale=config.weight_scale,
            noise_scale=config.noise_scale,
            nonlinearity=config.nonlinearity,
            intervention_type="soft_weight",
            weight_scale_intervened=config.weight_scale,
            alt_nonlinearity=None,
            shift_scale=2.0,
            noise_scale_intervened=None,
        )
        return df, true_target

    raise RuntimeError(f"Unknown data config: {config!r}")


def run_algo(df: pd.DataFrame, data_cfg: DataConfig, algo_cfg: AlgoConfig) -> Any:
    # context_col needed only for *_CONTEXTS modes
    context_col = getattr(data_cfg, "context_col", "context")

    # map algo name to a search strategy (can be extended later)
    if isinstance(algo_cfg, TopicAlgoConfig):
        graph_search = GraphSearch.TOPIC
    elif isinstance(algo_cfg, LincAlgoConfig):
        graph_search = GraphSearch.TOPIC
    elif isinstance(algo_cfg, SpaceTimeAlgoConfig):
        graph_search = GraphSearch.TOPIC
    elif isinstance(algo_cfg, SpaceTimeCAlgoConfig):
        graph_search = GraphSearch.TOPIC
    else:
        raise RuntimeError(f"Unknown algo config: {algo_cfg!r}")

    est = CausalChange(
        data_mode=data_cfg.setting,
        graph_search=graph_search,
        context_col=context_col,
        vb=0,
    )
    return est.fit(df)

def run_scoring(true_g, est_dag, scoring_cfg: ScoringConfig) -> dict[str, float]:
    # CausalChange returns a networkx.DiGraph already.
    m = compute_metrics(true_g, est_dag)

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

    # Avoid producing huge numbers of duplicate configs by only iterating
    # over parameters relevant to each DataMode.
    settings = data_grid.get("setting", [DataMode.IID])
    for setting in settings:
        try:
            mode = DataMode(setting)
        except Exception:
            mode = setting

        base_keys = {
            "setting": [mode],
            "linearity": data_grid.get("linearity", ["linear"]),
            "n_nodes": data_grid.get("n_nodes", [5]),
            "edge_prob": data_grid.get("edge_prob", [0.4]),
            "seed": data_grid.get("seed", [42]),
            "weight_scale": data_grid.get("weight_scale", [2.0]),
            "noise_scale": data_grid.get("noise_scale", [0.7]),
        }

        if mode == DataMode.IID:
            base_keys.update({
                "n_samples": data_grid.get("n_samples", [500]),
                "nonlinearity": data_grid.get("nonlinearity", ["tanh"]),
            })
        elif mode == DataMode.CONTEXTS:
            base_keys.update({
                "context_col": data_grid.get("context_col", ["context"]),
                "n_contexts": data_grid.get("n_contexts", [5]),
                "n_samples_per_context": data_grid.get("n_samples_per_context", [200]),
                "n_intervened_per_context": data_grid.get("n_intervened_per_context", [1]),
                "intervention_type": data_grid.get("intervention_type", ["soft_weight"]),
                "weight_scale_intervened": data_grid.get("weight_scale_intervened", [2.0]),
                "shift_scale": data_grid.get("shift_scale", [2.0]),
                "noise_scale_intervened": data_grid.get("noise_scale_intervened", [None]),
                "nonlinearity": data_grid.get("nonlinearity", ["tanh"]),
                "alt_nonlinearity": data_grid.get("alt_nonlinearity", [None]),
            })
        elif mode == DataMode.TIME:
            base_keys.update({
                "n_samples": data_grid.get("n_samples", [500]),
                "tau_max": data_grid.get("tau_max", [1]),
                "nonlinearity": data_grid.get("nonlinearity", ["tanh"]),
            })
        elif mode == DataMode.TIME_CONTEXTS:
            base_keys.update({
                "context_col": data_grid.get("context_col", ["context"]),
                "n_contexts": data_grid.get("n_contexts", [5]),
                "n_samples_per_context": data_grid.get("n_samples_per_context", [200]),
                "tau_max": data_grid.get("tau_max", [1]),
                "nonlinearity": data_grid.get("nonlinearity", ["tanh"]),
            })
        else:
            continue

        for data_opt0 in _product_dict(base_keys):
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
                elif name == "spacetime":
                    algo_opt = _filter_to_model_fields(SpaceTimeAlgoConfig, algo_opt0)
                elif name == "spacetime_c":
                    algo_opt = _filter_to_model_fields(SpaceTimeCAlgoConfig, algo_opt0)
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

    # grid may provide setting as a string ("iid", "contexts", ...) or as the Enum itself.
    try:
        mode = DataMode(setting)
    except Exception:
        mode = setting
    if mode == DataMode.IID and linearity == "linear":
        return SingleLinearDataConfig
    if mode == DataMode.IID and linearity == "nonlinear":
        return SingleNonlinearDataConfig
    if mode == DataMode.CONTEXTS and linearity == "linear":
        return MultiLinearDataConfig
    if mode == DataMode.CONTEXTS and linearity == "nonlinear":
        return MultiNonlinearDataConfig
    if mode == DataMode.TIME and linearity == "linear":
        return SingleTemporalLinearDataConfig
    if mode == DataMode.TIME and linearity == "nonlinear":
        return SingleTemporalNonlinearDataConfig
    if mode == DataMode.TIME_CONTEXTS and linearity == "linear":
        return MultiTemporalLinearDataConfig
    if mode == DataMode.TIME_CONTEXTS and linearity == "nonlinear":
        return MultiTemporalNonlinearDataConfig

    return None