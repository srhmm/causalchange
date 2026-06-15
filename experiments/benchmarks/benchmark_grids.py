GRID_TOPIC_SMALL = {
    "data": {
        "setting": ["single"],
        "n_nodes": [5],
        "edge_prob": [0.4],
        "n_samples": [1000],
        "nonlinearity": ["tanh", "sin", "lin", "relu"],  # better name would be funform
    },
    "algo": {
        "name": ["topic"],
        "score_type": ["gam"],
    },
    "scoring": {
        "metrics": [
            ["edge_f1", "skel_f1"],
        ]  # uncomment returns all impltd metrics
    },
}
GRID_TOPIC_MEDIUM = {
    "data": {
        "setting": ["single"],
        "n_nodes": [5, 10, 20],
        "edge_prob": [0.2, 0.4, 0.6],
        "n_samples": [1000],
        "nonlinearity": ["lin", "tanh", "sin", "relu"],
    },
    "algo": {
        "name": ["topic"],
        "score_type": ["lin", "gam", "spline"],
    },
}
GRID_LINC_NO_CHANGE = {
    "data": {
        "setting": ["multi"],
        "n_nodes": [5, 10],
        "edge_prob": [0.3],
        "n_contexts": [3, 5],
        "n_samples_per_context": [300, 700],
        "n_context_clusters": [1],
        "context_col": ["context"],
        "nonlinearity": ["lin", "tanh", "sin"],
        "mechanism_change_fraction": [0.0],
        "mechanism_shift_scale": [0.0],
    },
    "algo": {
        "name": ["linc"],
        "score_type": ["lin", "spline"],
    },
}
GRID_LINC_CONTEXT_CHANGES = {
    "data": {
        "setting": ["multi"],
        "n_nodes": [5, 10],
        "edge_prob": [0.3],
        "n_contexts": [4, 6],
        "n_samples_per_context": [300, 700],
        "n_context_clusters": [2, 3],
        "context_col": ["context"],
        "nonlinearity": ["lin", "tanh", "sin"],
        "mechanism_change_fraction": [0.25, 0.5, 0.75],
        "mechanism_shift_scale": [0.5, 1.0],
    },
    "algo": {
        "name": ["linc"],
        "score_type": ["lin", "spline"],
    },
}
GRID_SPACETIME_ORACLE_TIME = {
    "data": {
        "setting": ["time"],
        "n_nodes": [3, 5, 10],
        "edge_prob": [0.3],
        "n_samples": [500, 1000],
        "tau_max": [1, 2],
        "n_changepoints": [1, 2],
        "n_regimes": [2, 3],
        "min_segment_length": [80],
        "nonlinearity": ["lin", "tanh"],
        "mechanism_change_fraction": [0.25, 0.5],
        "mechanism_shift_scale": [0.75],
        "allow_self_lag": [True],
    },
    "algo": {
        "name": ["spacetime"],
        "score_type": ["lin"],
        "changepoint_mode": ["oracle"],
    },
}
GRID_SPACETIME_ORACLE_CONTEXTS = {
    "data": {
        "setting": ["time-contexts"],
        "n_nodes": [3, 5],
        "edge_prob": [0.3],
        "n_contexts": [3, 5],
        "n_samples_per_context": [500],
        "n_context_clusters": [1, 2],
        "context_col": ["context"],
        "tau_max": [1, 2],
        "n_changepoints": [1, 2],
        "n_regimes": [2, 3],
        "min_segment_length": [80],
        "nonlinearity": ["lin", "tanh"],
        "mechanism_change_fraction": [0.25, 0.5],
        "mechanism_shift_scale": [0.75],
        "allow_self_lag": [True],
    },
    "algo": {
        "name": ["spacetime"],
        "score_type": ["lin"],
        "changepoint_mode": ["oracle"],
    },
}
GRID_SPACETIME_DETECT_SMALL = {
    "data": {
        "setting": ["time", "time-contexts"],
        "n_nodes": [3, 5],
        "edge_prob": [0.3],
        "n_samples": [700],
        "n_contexts": [3],
        "n_samples_per_context": [700],
        "n_context_clusters": [1, 2],
        "context_col": ["context"],
        "tau_max": [1],
        "n_changepoints": [1, 2],
        "n_regimes": [2, 3],
        "min_segment_length": [100],
        "nonlinearity": ["lin"],
        "mechanism_change_fraction": [0.5],
        "mechanism_shift_scale": [1.0],
        "allow_self_lag": [True],
    },
    "algo": {
        "name": ["spacetime"],
        "score_type": ["lin"],
        "changepoint_mode": ["detect"],
    },
}