GRID_TOPIC_SMALL = {
    "data": {
        "setting": ["single"],
        "n_nodes": [5],
        "edge_prob": [0.4],
        "n_samples": [1000],
        "nonlinearity": ["tanh", "sin", "lin", "relu"],
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
        "n_intervened_per_context": [0],
        "context_col": ["context"],
        "nonlinearity": ["tanh", "sin", "lin", "relu"],
        "intervention_type": ["soft-weight"],
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
        "n_intervened_per_context": [1],
        "context_col": ["context"],
        "nonlinearity": ["tanh", "sin", "lin", "relu"],
        "intervention_type": ["soft-weight"],
        "weight_scale_intervened": [2.0],
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
        "n_intervened_per_context": [0],
        "context_col": ["context"],
        "nonlinearity": ["lin", "tanh", "sin"],
        "intervention_type": ["soft-weight"],
    },
    "algo": {
        "name": ["linc"],
        "score_type": ["lin", "gam", "spline"],
    },
}


GRID_LINC_CONTEXT_CHANGES = {
    "data": {
        "setting": ["multi"],
        "n_nodes": [5, 10],
        "edge_prob": [0.3],
        "n_contexts": [4, 6],
        "n_samples_per_context": [300, 700],
        "n_intervened_per_context": [1, 2],
        "context_col": ["context"],
        "nonlinearity": ["lin", "tanh", "sin"],
        "intervention_type": ["soft-weight", "shift", "noise"],
        "weight_scale_intervened": [2.0],
        "shift_scale": [1.0, 2.0],
        "noise_scale_intervened": [1.0],
    },
    "algo": {
        "name": ["linc"],
        "score_type": ["lin", "gam", "spline"],
    },
}


GRID_SPACETIME_SMALL_STATIONARY_TIME = {
    "data": {
        "setting": ["time"],
        "n_nodes": [5],
        "edge_prob": [0.3],
        "n_samples": [700],
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


GRID_SPACETIME_SMALL_STATIONARY_CONTEXTS = {
    "data": {
        "setting": ["time-contexts"],
        "n_nodes": [5],
        "edge_prob": [0.3],
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


GRID_SPACETIME_SMALL_FIXED_TIME = {
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
        "changepoint_mode": ["fixed"],
        "changepoint_scope": ["global"],
        "changepoint_method": ["skip"],
        "clustering_scope": ["skip"],
        "clustering_method": ["skip"],
        "testing_method": ["skip"],
    },
}


GRID_SPACETIME_SMALL_FIXED_CONTEXTS = {
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


GRID_SPACETIME_CLUSTERING_CONTEXTS_FIXED = {
    "data": {
        "setting": ["time-contexts"],
        "n_nodes": [5],
        "edge_prob": [0.3],
        "n_contexts": [3],
        "n_samples_per_context": [700],
        "n_context_clusters": [2],
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
        "clustering_scope": ["regimes-contexts"],
        "clustering_method": ["mechanism-clustering"],
        "testing_method": ["skip"],
    },
}
GRID_CMM_MIXED_SMALL = {
    "data": {
        "setting": ["mixed"],
        "n_nodes": [5],
        "edge_prob": [0.4],
        "n_mechanisms": [2],
        "n_samples_per_mechanism": [300],
        "n_mixed_variables": [1, 2],
        "cluster_mode": ["global", "local"],
        "nonlinearity": ["lin", "tanh"],
        "alt_nonlinearity": ["sin"],
        "mechanism_change": ["soft-weight"],
    },
    "algo": {
        "name": ["cmm"],
        "score_type": ["lin"],
        "mix_type": ["lin"],
        "k_max": [2],
        "lambda_mix": [0.0],
        "hybrid_mixing": [False],
    },
}

ALL_GRIDS = {
    # TOPIC
    "topic_small": GRID_TOPIC_SMALL,
    "topic_medium": GRID_TOPIC_MEDIUM,
    # LINC
    "linc_small_no_change": GRID_LINC_SMALL_NO_CHANGE,
    "linc_small_context_changes": GRID_LINC_SMALL_CONTEXT_CHANGES,
    "linc_no_change": GRID_LINC_NO_CHANGE,
    "linc_context_changes": GRID_LINC_CONTEXT_CHANGES,
    # SpaceTime
    "spacetime_small_stationary_time": GRID_SPACETIME_SMALL_STATIONARY_TIME,
    "spacetime_small_stationary_contexts": GRID_SPACETIME_SMALL_STATIONARY_CONTEXTS,
    "spacetime_small_fixed_time": GRID_SPACETIME_SMALL_FIXED_TIME,
    "spacetime_small_fixed_contexts": GRID_SPACETIME_SMALL_FIXED_CONTEXTS,
    "spacetime_small_detect_time": GRID_SPACETIME_SMALL_DETECT_TIME,
    "spacetime_small_detect_contexts": GRID_SPACETIME_SMALL_DETECT_CONTEXTS,
    "spacetime_clustering_contexts_fixed": GRID_SPACETIME_CLUSTERING_CONTEXTS_FIXED,
    # "cmm_small": GRID_CMM_MIXED_SMALL,
}


SELECTED_GRIDS = ["linc_small_no_change", "linc_small_context_changes"]


GRIDS = {name: ALL_GRIDS[name] for name in SELECTED_GRIDS}
