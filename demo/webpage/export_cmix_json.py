import json
import numpy as np
from pathlib import Path

from util import run_cmm_on_node_pair, load_colon_data

colon_df = load_colon_data()

x = colon_df.x3.to_numpy()
y = colon_df.y3.to_numpy()

model = run_cmm_on_node_pair(x, y, ret_model=True)


assignments = np.array(model.e_Z_n[1])
soft = np.array(model.e_Zp_n[1])
K = len(np.unique(assignments))

bic_XtoY, res_XtoY = model.edges_state.score_edge(0, [1])
bic_YtoX, res_YtoX = model.edges_state.score_edge(1, [0])



points = []
for i in range(len(x)):
    entry = {
        "x": float(x[i]),
        "y": float(y[i]),
        "component": int(assignments[i])
    }
    if soft is not None:
        entry["soft"] = [float(v) for v in soft[i]]
    points.append(entry)

data_out = {
    "points": points,
    "components": int(K),
    "bic": {
        "X_to_Y": float(bic_XtoY),
        "Y_to_X": float(bic_YtoX)
    },
    "meta": {
        "dataset": "colon_data",
        "variable_pair": "x3,y3"
    }
}

out_path = Path("cmix_example_x3_y3.json")
with out_path.open("w") as f:
    json.dump(data_out, f, indent=2)

print("Saved:", out_path)
