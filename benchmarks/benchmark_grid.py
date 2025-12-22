
BENCHMARK_GRID = {
    "data": {
        "setting": [
            "multi"],
        "linearity": ["linear", "nonlinear"],
        "n_nodes": [5],
        "edge_prob": [0.4],

        "n_samples": [500],

        "n_contexts": [10],
        "n_samples_per_context": [200],
        "n_intervened_per_context": [2],
        "context_col": ["context"],
        "intervention_type": ["hard", "soft_weight", "shift", "noise", "soft_mechanism"],

        "nonlinearity": ["tanh"], #, "sin"],
        "alt_nonlinearity": ["sin"],
    },
    "algo": {
        "name": ["linc", "topic" ],
        "score_type": ["gam"], # "aic-g",
        "context_col": ["context"],
    },

    "scoring": {
        "metrics": [
            ["edge_f1", "skel_f1"],  # ["shd", "edge_f1", "skel_f1", "time_s"],
        ]
    }
}