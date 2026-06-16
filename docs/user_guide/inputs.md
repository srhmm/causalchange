### Inputs

---

### `Topic`

| Parameter             |                                         Type / values |  Default | Description                                                     |
| --------------------- | ----------------------------------------------------: | -------: | --------------------------------------------------------------- |
| `score_type`          | `"lin"`, `"gam"`, `"spline"`, `"krr"`, `"gp"`, `"ff"` |  `"gam"` | Local regression/scoring model used for local causal scores.    |
| `postprocessing_mode` |                          `"skip"`, `"edge-strengths"` | `"skip"` | Optional postprocessing after graph discovery.                  |
| `score_kwargs`        |                              `dict[str, Any] \| None` |   `None` | Additional keyword arguments passed to the local scoring model. |
| `seed`                |                                                 `int` |     `42` | Random seed used by stochastic scoring components.              |
| `var_nms`             |                                   `list[str] \| None` |   `None` | Optional variable names for display, debugging, and plotting.   |

---

### `Linc`

| Parameter                    |                                         Type / values |     Default | Description                                                               |
| ---------------------------- | ----------------------------------------------------: | ----------: | ------------------------------------------------------------------------- |
| `score_type`                 | `"lin"`, `"gam"`, `"spline"`, `"krr"`, `"gp"`, `"ff"` |     `"gam"` | Local regression/scoring model used for local causal scores.              |
| `context_col`                |                                                 `str` | `"context"` | Name of the observed context indicator column.                            |
| `postprocessing_mode`        |                          `"skip"`, `"edge-strengths"` |    `"skip"` | Optional postprocessing after graph discovery.                            |
| `context_combination_kwargs` |                    `ContextCombinationKwargs \| None` |      `None` | Optional parameters controlling how context-specific scores are combined. |
| `score_kwargs`               |                              `dict[str, Any] \| None` |      `None` | Additional keyword arguments passed to the local scoring model.           |
| `seed`                       |                                                 `int` |        `42` | Random seed used by stochastic scoring components.                        |
| `var_nms`                    |                                   `list[str] \| None` |      `None` | Optional variable names for display, debugging, and plotting.             |

---

### `CMM`

| Parameter             |                                               Type / values |  Default | Description                                                                                                        |
| --------------------- | ----------------------------------------------------------: | -------: | ------------------------------------------------------------------------------------------------------------------ |
| `mix_type`            | `"lin"`, `"quadratic"`, `"cubic"`, `"nspline"`, `"bspline"` |  `"lin"` | Mixture-regression family used by the CMM local score.                                                             |
| `k_max`               |                                              positive `int` |      `5` | Maximum number of mixture components considered. Candidate values `1, ..., k_max` are compared by model selection. |
| `postprocessing_mode` |                                `"skip"`, `"edge-strengths"` | `"skip"` | Optional postprocessing after graph discovery.                                                                     |
| `score_kwargs`        |                                    `dict[str, Any] \| None` |   `None` | Additional CMM scoring arguments, for example `degree` for spline mixture terms.                                   |
| `seed`                |                                                       `int` |     `42` | Random seed used by stochastic scoring components.                                                                 |
| `var_nms`             |                                         `list[str] \| None` |   `None` | Optional variable names for display, debugging, and plotting.                                                      |

`CMM` fixes the score convention internally and does not expose `score_type`. Mixture-regression shape is controlled by `mix_type`.

Conditional mixtures with parents require the optional R/rpy2/flexmix dependency stack. Parentless mechanisms use a sklearn Gaussian mixture fallback.

---

### `SpaceTime`

| Parameter              |                                               Type / values |                 Default | Description                                                                       |
| ---------------------- | ----------------------------------------------------------: | ----------------------: | --------------------------------------------------------------------------------- |
| `score_type`           |       `"lin"`, `"gam"`, `"spline"`, `"krr"`, `"gp"`, `"ff"` |                 `"gam"` | Local regression/scoring model used for local causal scores.                      |
| `data_mode`            |                                 `"time"`, `"time-contexts"` |       `"time-contexts"` | Whether the input is a single time series or multiple time-series contexts.       |
| `tau_max`              |                                              positive `int` |                     `2` | Maximum time lag considered for temporal causal edges.                            |
| `context_col`          |                                                       `str` |             `"context"` | Name of the context indicator column when `data_mode="time-contexts"`.            |
| `changepoint_mode`     |                             `"skip"`, `"fixed"`, `"detect"` |              `"detect"` | Whether to skip changepoints, use fixed changepoints, or detect changepoints.     |
| `changepoint_scope`    |                       `"skip"`, `"global"`, `"per-context"` |              `"global"` | Whether changepoints are shared globally or detected per context.                 |
| `changepoint_method`   |                                          `"skip"`, `"pelt"` |                `"pelt"` | Changepoint detection algorithm.                                                  |
| `clustering_scope`     |   `"skip"`, `"regimes"`, `"contexts"`, `"regimes-contexts"` |    `"regimes-contexts"` | Scope over which causal mechanisms may be clustered.                              |
| `clustering_method`    | `"skip"`, `"statistical-testing"`, `"mechanism-clustering"` | `"statistical-testing"` | Method used to compare or cluster causal mechanisms.                              |
| `testing_method`       |                              `"skip"`, `"kernel"`, `"none"` |              `"kernel"` | Mechanism testing method when `clustering_method="statistical-testing"`.          |
| `d_min`                |                                              positive `int` |                    `30` | Minimum segment/window length used for changepoint detection and temporal search. |
| `max_iter`             |                                              positive `int` |                     `3` | Maximum number of SpaceTime search iterations.                                    |
| `pelt_penalty`         |               positive `float`, `"auto"`, `"bic"`, `"mbic"` |                `"auto"` | Penalty used by PELT changepoint detection.                                       |
| `mechanism_test_alpha` |                                         `float` in `(0, 1)` |                  `0.05` | Significance level for mechanism equality tests.                                  |
| `fixed_changepoints`   |                                         `list[int] \| None` |                  `None` | Fixed changepoints used when `changepoint_mode="fixed"`.                          |
| `postprocessing_mode`  |                                `"skip"`, `"edge-strengths"` |                `"skip"` | Optional postprocessing after graph discovery.                                    |
| `score_kwargs`         |                                    `dict[str, Any] \| None` |                  `None` | Additional keyword arguments passed to the local scoring model.                   |
| `seed`                 |                                                       `int` |                    `42` | Random seed used by stochastic scoring components.                                |
| `var_nms`              |                                         `list[str] \| None` |                  `None` | Optional variable names for display, debugging, and plotting.                     |

When `changepoint_mode="skip"`, also pass `changepoint_scope="skip"` and `changepoint_method="skip"`.

When mechanism clustering is skipped, also pass `clustering_scope="skip"`, `clustering_method="skip"`, and `testing_method="skip"`.

For detected changepoints with `changepoint_method="pelt"`, install the optional changepoint dependency. For kernel-based mechanism testing with `testing_method="kernel"`, install the optional testing dependency.
