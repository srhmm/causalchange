import json
import numpy as np
import pandas as pd
from pathlib import Path

from util import run_topic_on_data, load_synthetic_data, replace_nans_with_none

for seed in range(1):
    n_nodes = 5
    X, truths = load_synthetic_data(seed=seed, n_nodes=n_nodes)


    G_hat, cc = run_topic_on_data(X, ret_model=True)

    print("Estimated", G_hat.edges)

    print("True", truths['true_g'].edges)

    if 'true_g' in truths:
        true_edges = [
            {"from": int(u), "to": int(v)}
            for u, v in truths['true_g'].edges()
        ]
    else:
        true_edges = []

    history_out = {
        "nodes": list(range(cc.N)),
        "node_names": cc.node_nms,
        "steps": cc.search_history,
        "true_edges": true_edges,
    }

    history_clean = replace_nans_with_none(history_out)

    out_path = Path(f"topic_history_synthetic_n{n_nodes}.json")

    out_path.write_text(json.dumps(history_clean, indent=2))




