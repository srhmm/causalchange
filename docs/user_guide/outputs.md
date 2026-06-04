### Outputs

___

After calling

```python
cc.fit(X)
```

General Results

| Attribute | Meaning | Availability                                                      |
|---|---|-------------------------------------------------------------------|
| `result_` | Full result object | all                                                        |
| `graph_` | Learned graph | all                                                         |
| `topological_order_` | Learned topological order, if available | TOPIC-style searches                                              |
| `history_` | Search or iteration history | all                                                               |
| `edge_strengths_` | Edge contribution scores | Only when `postprocessing_mode=PostprocessingMode.EDGE_STRENGTHS` |
| `result_.diagnostics` | Additional diagnostics | all                                                         |

Temporal Results

| Attribute | Meaning |
|---|---|
| `changepoints_` | Detected or fixed changepoints |
| `changepoints_by_context_` | Per-context changepoints when `changepoint_scope=ChangepointScope.PER_CONTEXT` |
| `partitions_` | Context/regime mechanism partition result |
| `result_.changepoint_diagnostics` | Changepoint detection diagnostics |
| `result_.grid_clusters` | Same partition object as `partitions_` |
