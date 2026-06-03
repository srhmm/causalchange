### Outputs

___

After calling

```python
cc.fit(X)
```

fitted attributes include

| Attribute            | Meaning                                 |
| -------------------- | --------------------------------------- |
| `graph_`             | learned graph                           |
| `result_`            | full result object                      |
| `edge_strengths_`    | edge contribution scores                |
| `topological_order_` | learned topological order, if available |
| `history_`           | search or iteration history             |
| `diagnostics_`       | additional diagnostics                  |

Temporal models also expose

| Attribute                  | Meaning                               |
| -------------------------- | ------------------------------------- |
| `changepoints_`            | detected or fixed changepoints        |
| `changepoints_by_context_` | per-context changepoints              |
| `grid_clusters_`           | context and regime mechanism clusters |
| `changepoint_diagnostics_` | changepoint detection diagnostics     |
