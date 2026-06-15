from experiments.benchmarks.run_methods import iter_valid_configs, run_on_config


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
        "algo": {
            "name": ["spacetime"],
            "score_type": ["lin"],
            "changepoint_mode": ["none"],
        },
    }

    cfg = next(iter_valid_configs(grid))
    metrics = run_on_config(cfg)

    assert "summary_shd" in metrics
    assert "wcg_shd" in metrics
    assert "time_s" in metrics