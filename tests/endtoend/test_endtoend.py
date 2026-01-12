import networkx as nx
import pytest

from benchmarks.run_methods import run_on_config
from causalchange.config.benchmark_config import BenchmarkConfig
from causalchange.config._cc_types import DataMode, GraphSearch, ScoreType, ContextAggregation


@pytest.mark.parametrize(
    "data_mode",
    [DataMode.IID, DataMode.CONTEXTS, DataMode.MIXED, DataMode.TIME, DataMode.TIME_CONTEXTS],
)
@pytest.mark.parametrize(
    "graph_search",
    [GraphSearch.TOPIC]
)

@pytest.mark.parametrize(
    "context_aggregation",
    [ContextAggregation.CHAIN, ContextAggregation.SKIP]
)
@pytest.mark.parametrize(
    "score_type",
    [ScoreType.LIN]
)
def test_end_to_end(data_mode, graph_search, score_type, context_aggregation):
    if not graph_search.is_compatible_with(data_mode) or not context_aggregation.is_compatible_with(data_mode):
        pytest.skip(f"{graph_search} not compatible with data_mode {data_mode}")

    if data_mode == DataMode.MIXED:
        pytest.skip(f"Data mode {DataMode.MIXED} add algo mixin")

    if data_mode in [DataMode.TIME, DataMode.TIME_CONTEXTS]:
        pytest.skip(f"Data mode {data_mode}")

    test_fun = _test_e2e_single if data_mode == DataMode.IID \
        else _test_e2e_mixed if data_mode == DataMode.MIXED \
        else _test_e2e_multi if data_mode == DataMode.CONTEXTS \
        else _test_e2e_time if data_mode == DataMode.TIME \
        else _test_e2e_time_contexts if data_mode == DataMode.TIME_CONTEXTS  else None
    assert test_fun is not None,  f"Data mode {data_mode}"

    cfg = _get_config_for_data_and_algo(data_mode, graph_search, score_type, context_aggregation)
    test_fun(cfg)



def _test_e2e_mixed(cfg: BenchmarkConfig):
    pytest.importorskip("rpy2", reason="MIXED mode requires rpy2")

    metrics, est_g = run_on_config(cfg, return_nx=True)
    assert isinstance(est_g, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(est_g)
    assert all(u != v for u, v in est_g.edges)
    #assert metrics["skel_f1"] != 0

def _test_e2e_single(cfg: BenchmarkConfig):
    metrics, est_g = run_on_config(cfg, return_nx=True)

    assert isinstance(est_g, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(est_g)
    assert all(u != v for u, v in est_g.edges)
    #assert metrics["skel_f1"] != 0


def _test_e2e_multi(cfg: BenchmarkConfig):
    metrics, est_g = run_on_config(cfg, return_nx=True)

    assert isinstance(est_g, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(est_g)
    assert all(u != v for u, v in est_g.edges)
    #assert metrics["skel_f1"] != 0


def _test_e2e_time(cfg: BenchmarkConfig):
    metrics, est_g = run_on_config(cfg, return_nx=True)
    assert isinstance(est_g, nx.DiGraph)
    #assert metrics["skel_f1"] != 0



def _test_e2e_time_contexts(cfg: BenchmarkConfig):
    metrics, est_g = run_on_config(cfg, return_nx=True)
    assert isinstance(est_g, nx.DiGraph)
    #assert metrics["skel_f1"] != 0



#DEFAULT_SCORING = {} #"metrics": ["edge_f1", "skel_f1"]}


def _get_config_for_data_and_algo(data_mode: DataMode, graph_search: GraphSearch, score_type: ScoreType, context_aggregation = ContextAggregation.SKIP) -> BenchmarkConfig:
    assert  graph_search.is_compatible_with(data_mode), f"{graph_search} not compatible with {data_mode}"

    setting = str(data_mode.value)

    if data_mode in (DataMode.TIME, DataMode.TIME_CONTEXTS):
        algo_name = "spacetime" if graph_search == GraphSearch.TOPIC else "spacetime-globe"
        if data_mode == DataMode.TIME_CONTEXTS:
            algo_name = "spacetime-c" if graph_search == GraphSearch.TOPIC else "spacetime-globe-c"
    elif data_mode == DataMode.CONTEXTS:
        algo_name = "linc" if context_aggregation == ContextAggregation.LINC else "chain" if context_aggregation == ContextAggregation.CHAIN else None
    else:
        algo_name = "topic" if graph_search == GraphSearch.TOPIC else "globe"


    data_payload: dict = {
        "setting": setting,
        "n_nodes": 5,
        "edge_prob": 0.4,
        "seed": 43,
        "weight_scale": 2.0,
        "noise_scale": 0.7,
        "nonlinearity": "lin",
    }

    if setting == "single": data_payload.update({"n_samples": 500})
    elif setting == "multi" or setting == "mixed":
        data_payload.update({
            "context_col": "context",
            "n_contexts": 3,
            "n_samples_per_context": 200,
            "n_intervened_per_context": 1,
            "intervention_type": "soft_weight",
            "alt_nonlinearity": "sin",
        })
    elif setting == "time": data_payload.update({"n_samples": 400, "tau_max": 1})
    elif setting == "time-contexts":
        data_payload.update({
            "context_col": "context",
            "n_contexts": 3,
            "n_samples_per_context": 200,
            "tau_max": 1,
            "n_intervened_per_context": 1,
            "intervention_type": "soft_weight",
        })

    cfg_dict = {
        "data": data_payload,
        "algo": {
            "name": algo_name,
            "score_type": score_type.value.lower() if hasattr(score_type, "value") else str(score_type),
            **({"context_col": "context"} if setting in ("multi", "time_contexts") else {}),
        },
        "scoring": {} #DEFAULT_SCORING,
    }

    return BenchmarkConfig.model_validate(cfg_dict)
