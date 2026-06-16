### Outputs

---

### General results

| Attribute             | Meaning                                                     | Availability                                |
| --------------------- | ----------------------------------------------------------- | ------------------------------------------- |
| `result_`             | Full fitted result object                                   | all algorithms                              |
| `graph_`              | Learned graph                                               | all algorithms                              |
| `topological_order_`  | Learned topological order, if produced by the graph search  | TOPIC-style searches                        |
| `history_`            | Search or iteration history                                 | all algorithms                              |
| `edge_strengths_`     | Edge contribution scores                                    | when `postprocessing_mode="edge-strengths"` |
| `result_.diagnostics` | Additional diagnostics                                      | all algorithms                              |
| `public_config_`      | Validated internal configuration used by the public wrapper | all public wrappers                         |

---

### Tabular results

| Attribute              | Meaning                                      | Availability                                |
| ---------------------- | -------------------------------------------- | ------------------------------------------- |
| `graph_`               | Learned tabular causal graph                 | `Topic`, `Linc`, `CMM`                      |
| `topological_order_`   | Learned topological order                    | `Topic`, `Linc`, `CMM`                      |
| `history_`             | Graph-search history                         | `Topic`, `Linc`, `CMM`                      |
| `edge_strengths_`      | Optional edge-strength postprocessing result | when `postprocessing_mode="edge-strengths"` |
| `result_.graph_search` | Full graph-search result object              | `Topic`, `Linc`, `CMM`                      |

---

### CMM results

| Attribute                                                         | Meaning                                                         |
| ----------------------------------------------------------------- | --------------------------------------------------------------- |
| `mixture_components_`                                             | Final-graph mixture assignments and responsibilities per target |
| `cmm_components_`                                                 | Alias for `mixture_components_`                                 |
| `cmm_labels_`                                                     | Hard component labels per target                                |
| `mixture_components_.target_components[target].labels`            | Hard component assignment for each row for one target mechanism |
| `mixture_components_.target_components[target].responsibilities`  | Posterior component probabilities for each row and component    |
| `mixture_components_.target_components[target].component_weights` | Estimated component weights for one target mechanism            |
| `mixture_components_.target_components[target].parents`           | Final parent set used for that target mechanism                 |
| `mixture_components_.target_components[target].score`             | Final local mixture score for that target mechanism             |
| `mixture_components_.target_components[target].n_components`      | Number of selected mixture components for that target mechanism |
| `mixture_components_.target_components[target].diagnostics`       | Target-level mixture diagnostics                                |
| `mixture_components_.global_labels`                               | Optional global component labels, if available                  |
| `mixture_components_.global_responsibilities`                     | Optional global component responsibilities, if available        |
| `mixture_components_.diagnostics`                                 | Overall CMM mixture extraction diagnostics                      |

---

### Temporal results

| Attribute                         | Meaning                                                         |
| --------------------------------- | --------------------------------------------------------------- |
| `changepoints_`                   | Detected or fixed changepoints                                  |
| `changepoints_by_context_`        | Per-context changepoints when `changepoint_scope="per-context"` |
| `changepoint_diagnostics_`        | Changepoint detection diagnostics                               |
| `partitions_`                     | Context/regime mechanism partition result                       |
| `result_.changepoint`             | Full changepoint result object                                  |
| `result_.changepoint.diagnostics` | Changepoint detection diagnostics in the result object          |
| `result_.mechanism_clustering`    | Mechanism clustering / partition result object                  |
| `result_.diagnostics`             | Temporal engine diagnostics                                     |
