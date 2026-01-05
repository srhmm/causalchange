
BENCHMARK_GRID = {
    "data": {
        "setting": [
            #"time",
           # "time-contexts",
            # "single",
             "multi"
                     ],
        "n_nodes": [5],
        "edge_prob": [0.4],
        "n_samples": [500],
        "n_contexts": [5],
        "n_samples_per_context": [500],
        "n_intervened_per_context": [3],
        "context_col": ["context"],
        "intervention_type": ["hard", "soft_weight", "shift", "noise"] , #, "soft_mechanism"],

        "nonlinearity": ["tanh", "sin", "lin", "relu"],
        "alt_nonlinearity": ["sin"],
    },
    "algo": {
        "name": [ "linc",
            #"chain",
                 # "topic" ,
             # "spacetime"  ,
            # "spacetime-c"
        ],
        "score_type": ["gam"],
        "context_col": ["context"],
        "tau_max": [2],
    },

    "scoring": {
        "metrics": [
            ["edge_f1", "skel_f1"],
        ]
    }
}