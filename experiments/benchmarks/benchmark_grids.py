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
GRID_LINC_SMALL_NO_CHANGE = {
    "data": {
        "setting": ["multi"],
        "n_nodes": [5],
        "edge_prob": [0.4],
        "n_contexts": [5],
        "n_samples_per_context": [700],
        "n_context_clusters": [1],
        "context_col": ["context"],
        "nonlinearity": ["tanh", "sin", "lin", "relu"],
        "mechanism_change_fraction": [0.0],
        "mechanism_shift_scale": [0.0],
    },
    "algo": {
        "name": ["linc"],
        "score_type": ["gam"],
    },
}
GRID_LINC_SMALL_CONTEXT_CHANGES = {
    "data": {
        "setting": ["multi"],
        "n_nodes": [5],
        "edge_prob": [0.4],
        "n_contexts": [5, 10],
        "n_samples_per_context": [700],
        "n_context_clusters": [3],
        "context_col": ["context"],
        "nonlinearity": ["tanh", "sin", "lin", "relu"],
        "mechanism_change_fraction": [0.5],
        "mechanism_shift_scale": [0.75],
    },
    "algo": {
        "name": ["linc"],
        "score_type": ["gam"],
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
}
GRID_SPACETIME_SMALL_STATIONARY = {
    "data": {
        "setting": ["time", "time-contexts"],
        "n_nodes": [5],
        "edge_prob": [0.3],
        "n_samples": [700],
        "n_contexts": [3],
        "n_samples_per_context": [700],
        "n_context_clusters": [1],
        "context_col": ["context"],
        "tau_max": [1],
        "n_changepoints": [0],
        "n_regimes": [1],
        "min_segment_length": [100],
        "nonlinearity": ["lin", "tanh"],
        "mechanism_change_fraction": [0.0],
        "mechanism_shift_scale": [0.0],
        "allow_self_lag": [True],
    },
    "algo": {
        "name": ["spacetime"],
        "score_type": ["lin"],
        "changepoint_mode": ["skip"],
        "changepoint_scope": ["skip"],
        "changepoint_method": ["skip"],
        "clustering_scope": ["skip"],
        "clustering_method": ["skip"],
        "testing_method": ["skip"],
    },
}
GRID_SPACETIME_SMALL_FIXED = {
    "data": {
        "setting": ["time"],
        "n_nodes": [5],
        "edge_prob": [0.3],
        "n_samples": [700],
        "n_contexts": [3],
        "n_samples_per_context": [700],
        "n_context_clusters": [1, 2],
        "context_col": ["context"],
        "tau_max": [1],
        "n_changepoints": [1],
        "n_regimes": [2],
        "min_segment_length": [100],
        "nonlinearity": ["lin", "tanh"],
        "mechanism_change_fraction": [0.5],
        "mechanism_shift_scale": [0.75],
        "allow_self_lag": [True],
    },
    "algo": {
        "name": ["spacetime"],
        "score_type": ["lin"],
        "changepoint_mode": ["fixed"],
        "changepoint_scope": ["global"],
        "changepoint_method": ["skip"],
        "clustering_scope": ["skip"],
        "clustering_method": ["skip"],
        "testing_method": ["skip"],
    },
}
GRID_SPACETIME_SMALL_DETECT_TIME = {
    "data": {
        "setting": ["time"],
        "n_nodes": [5],
        "edge_prob": [0.3],
        "n_samples": [700],
        "tau_max": [1],
        "n_changepoints": [1],
        "n_regimes": [2],
        "min_segment_length": [100],
        "nonlinearity": ["lin", "tanh"],
        "mechanism_change_fraction": [0.5],
        "mechanism_shift_scale": [0.75],
        "allow_self_lag": [True],
    },
    "algo": {
        "name": ["spacetime"],
        "score_type": ["lin"],
        "changepoint_mode": ["detect"],
        "changepoint_scope": ["global"],
        "changepoint_method": ["pelt"],
        "clustering_scope": ["skip"],
        "clustering_method": ["skip"],
        "testing_method": ["skip"],
    },
}
GRID_SPACETIME_SMALL_DETECT_CONTEXTS = {
    "data": {
        "setting": ["time-contexts"],
        "n_nodes": [5],
        "edge_prob": [0.3],
        "n_contexts": [3],
        "n_samples_per_context": [700],
        "n_context_clusters": [1, 2],
        "context_col": ["context"],
        "tau_max": [1],
        "n_changepoints": [1],
        "n_regimes": [2],
        "min_segment_length": [100],
        "nonlinearity": ["lin", "tanh"],
        "mechanism_change_fraction": [0.5],
        "mechanism_shift_scale": [0.75],
        "allow_self_lag": [True],
    },
    "algo": {
        "name": ["spacetime"],
        "score_type": ["lin"],
        "changepoint_mode": ["detect"],
        "changepoint_scope": ["global"],
        "changepoint_method": ["pelt"],
        "clustering_scope": ["skip"],
        "clustering_method": ["skip"],
        "testing_method": ["skip"],
    },
}

GRIDS = {
    # TOPIC
    #"topic_small": GRID_TOPIC_SMALL,
    #"topic_medium": GRID_TOPIC_MEDIUM,

    # LINC
    #"linc_small_no_change": GRID_LINC_SMALL_NO_CHANGE,
    #"linc_small_context_changes": GRID_LINC_SMALL_CONTEXT_CHANGES,
    #"linc_no_change": GRID_LINC_NO_CHANGE,
    #"linc_context_changes": GRID_LINC_CONTEXT_CHANGES,

    # SpaceTime
    #"spacetime_small_stationary": GRID_SPACETIME_SMALL_STATIONARY,
    #"spacetime_small_fixed": GRID_SPACETIME_SMALL_FIXED,
    "spacetime_small_detect_time": GRID_SPACETIME_SMALL_DETECT_TIME,
    "spacetime_small_detect_contexts": GRID_SPACETIME_SMALL_DETECT_CONTEXTS,
}
