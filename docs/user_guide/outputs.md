### Outputs

___

General Results


| Attribute | Meaning | Availability |
|---|---|---|
| `result_` | Full result object | all algorithms |
| `graph_` | Learned graph | all algorithms |
| `topological_order_` | Learned topological order, if available | TOPIC-style searches |
| `history_` | Search or iteration history | all algorithms |
| `edge_strengths_` | Edge contribution scores | when `postprocessing_mode="edge-strengths"` |
| `result_.diagnostics` | Additional diagnostics | all algorithms |
                                                   |

Temporal Results

| Attribute | Meaning |
|---|---|
| `changepoints_` | Detected or fixed changepoints |
| `changepoints_by_context_` | Per-context changepoints when `changepoint_scope="per-context"` |
| `partitions_` | Context/regime mechanism partition result |
| `result_.changepoint_diagnostics` | Changepoint detection diagnostics |
| `result_.grid_clusters` | Same partition object as `partitions_` |
