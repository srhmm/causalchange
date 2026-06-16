from config.benchmark_config import BenchmarkConfig

from experiments.benchmarks.run import iter_valid_configs, run_on_config


def test_benchmark_grid_single_topic_smoke():
    grid = {
        "data": {
            "setting": ["single"],
            "n_nodes": [3],
            "edge_prob": [0.4],
            "n_samples": [100],
            "nonlinearity": ["lin"],
        },
        "algo": {
            "name": ["topic"],
            "score_type": ["lin"],
        },
    }

    cfg = next(iter_valid_configs(grid))
    metrics = run_on_config(cfg)

    assert "shd" in metrics
    assert "time_s" in metrics


def test_benchmark_grid_spacetime_smoke_no_changepoints():
    grid = {
        "data": {
            "setting": ["time"],
            "n_nodes": [3],
            "edge_prob": [0.4],
            "n_samples": [120],
            "tau_max": [1],
            "n_changepoints": [0],
            "n_regimes": [1],
            "min_segment_length": [30],
            "nonlinearity": ["lin"],
            "mechanism_change_fraction": [0.0],
            "mechanism_shift_scale": [0.0],
            "allow_self_lag": [True],
        },
        "algo": {"name": ["spacetime"], "score_type": ["lin"], "changepoint_mode": ["skip"]},
    }

    cfg = next(iter_valid_configs(grid))
    metrics = run_on_config(cfg)

    assert "summary_shd" in metrics
    assert "wcg_shd" in metrics
    assert "time_s" in metrics


def test_cmm_mixed_benchmark_reports_per_target_mixture_metrics():
    cfg = BenchmarkConfig.model_validate(
        {
            "data": {
                "setting": "mixed",
                "n_nodes": 4,
                "edge_prob": 0.4,
                "n_mechanisms": 2,
                "n_samples_per_mechanism": 40,
                "n_mixed_variables": 2,
                "cluster_mode": "local",
                "nonlinearity": "lin",
                "mechanism_change": "soft-weight",
                "seed": 0,
            },
            "algo": {
                "name": "cmm",
                "score_type": "lin",
                "mix_type": "lin",
                "k_max": 2,
                "lambda_mix": 0.0,
                "hybrid_mixing": False,
                "max_em_iter": 10,
                "n_init": 1,
                "tol": 1e-4,
                "ridge": 1e-8,
            },
        }
    )

    metrics = run_on_config(cfg)

    assert "cmm_mixture_ari" in metrics
    assert "cmm_mixture_ami" in metrics
    assert "cmm_mixture_nmi" in metrics
    assert "cmm_mixture_n_targets" in metrics
    assert metrics["cmm_mixture_n_targets"] == 2.0

    per_target_ari = [key for key in metrics if key.startswith("cmm_mixture_X") and key.endswith("_ari")]
    per_target_ami = [key for key in metrics if key.startswith("cmm_mixture_X") and key.endswith("_ami")]

    assert len(per_target_ari) == 2
    assert len(per_target_ami) == 2
