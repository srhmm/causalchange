
BENCHMARK_GRID = {
    "data": {
        "setting": ["iid", "contexts", "time", "time-contexts"],
        "linearity": ["linear", "nonlinear"],
        "n_nodes": [5],
        "edge_prob": [0.4],

        "n_samples": [500],
        "tau_max": [1, 2],

        "n_contexts": [10],
        "n_samples_per_context": [200],
        "n_intervened_per_context": [2],
        "context_col": ["context"],
        "intervention_type": ["hard", "soft_weight", "shift", "noise", "soft_mechanism"],

        "nonlinearity": ["tanh"],
        "alt_nonlinearity": ["sin"],
    },
    "algo": {
        "name": ["topic", "linc", "spacetime", "spacetime_c"],
        "scoring_method": ["bic-g"],
        # LINC hyperparameters
        "context_col": ["context"],
    },

    "scoring": {
        "metrics": [
            ["edge_f1", "skel_f1"],
        ]
    }
}