### Inputs

---

### `Topic`

| Parameter             |                                         Type / values |  Default | Description                                                |
| --------------------- | ----------------------------------------------------: | -------: | ---------------------------------------------------------- |
| `score_type`          | `"lin"`, `"gam"`, `"spline"`, `"krr"`, `"gp"`, `"ff"` |  `"gam"` | Scoring or regression model used for local causal scores.  |
| `postprocessing_mode` |                          `"skip"`, `"edge-strengths"` | `"skip"` | Optional postprocessing after graph discovery.             |
| `score_kwargs`        |                              `dict[str, Any] \| None` |   `None` | Additional keyword arguments passed to the scoring method. |

### `Linc`

| Parameter             |                                         Type / values |     Default | Description                                                        |
| --------------------- | ----------------------------------------------------: | ----------: | ------------------------------------------------------------------ |
| `score_type`          | `"lin"`, `"gam"`, `"spline"`, `"krr"`, `"gp"`, `"ff"` |     `"gam"` | Scoring or regression model used for local causal scores.          |
| `context_col`         |                                                 `str` | `"context"` | Name of the column in `X` that identifies the context of each row. |
| `postprocessing_mode` |                          `"skip"`, `"edge-strengths"` |    `"skip"` | Optional postprocessing after graph discovery.                     |
| `score_kwargs`        |                              `dict[str, Any] \| None` |      `None` | Additional keyword arguments passed to the scoring method.         |

### `SpaceTime`

| Parameter              |                                               Type / values |                 Default | Description                                                                 |
| ---------------------- | ----------------------------------------------------------: | ----------------------: | --------------------------------------------------------------------------- |
| `data_mode`            |                                 `"time"`, `"time-contexts"` |       `"time-contexts"` | Whether the input is a single time series or multiple time-series contexts. |
| `score_type`           |       `"lin"`, `"gam"`, `"spline"`, `"krr"`, `"gp"`, `"ff"` |                 `"gam"` | Scoring or regression model used for local causal scores.                   |
| `tau_max`              |                                              positive `int` |                     `2` | Maximum time lag considered for temporal causal edges.                      |
| `context_col`          |                                                       `str` |             `"context"` | Name of the context column, required when `data_mode="time-contexts"`.      |
| `changepoint_mode`     |                             `"skip"`, `"fixed"`, `"detect"` |              `"detect"` | Whether to skip changepoints, use known changepoints, or detect them.       |
| `changepoint_scope`    |                       `"skip"`, `"global"`, `"per-context"` |              `"global"` | Whether changepoints are shared globally or detected per context.           |
| `changepoint_method`   |                                          `"skip"`, `"pelt"` |                `"pelt"` | Changepoint detection algorithm.                                            |
| `fixed_changepoints`   |                                         `list[int] \| None` |                  `None` | Known changepoints, used with `changepoint_mode="fixed"`.                   |
| `d_min`                |                                              positive `int` |                    `30` | Minimum segment/window length for changepoint detection.                    |
| `max_iter`             |                                              positive `int` |                     `3` | Maximum number of alternating changepoint/discovery iterations.             |
| `pelt_penalty`         |               positive `float`, `"auto"`, `"bic"`, `"mbic"` |                `"auto"` | Penalty used by PELT changepoint detection.                                 |
| `clustering_scope`     |   `"skip"`, `"regimes"`, `"contexts"`, `"regimes-contexts"` |    `"regimes-contexts"` | Scope over which causal mechanisms may be clustered.                        |
| `clustering_method`    | `"skip"`, `"statistical-testing"`, `"mechanism-clustering"` | `"statistical-testing"` | Method used to cluster or compare causal mechanisms.                        |
| `testing_method`       |                              `"skip"`, `"kernel"`, `"none"` |              `"kernel"` | Statistical test used when `clustering_method="statistical-testing"`.       |
| `mechanism_test_alpha` |                                         `float` in `(0, 1)` |                  `0.05` | Significance level for mechanism equality tests.                            |
| `postprocessing_mode`  |                                `"skip"`, `"edge-strengths"` |                `"skip"` | Optional postprocessing after graph discovery.                              |
| `score_kwargs`         |                                    `dict[str, Any] \| None` |                  `None` | Additional keyword arguments passed to the scoring method.                  |


### `CMM`

| Parameter             | Type / values                                             | Default | Description                                                     |
| --------------------- | ---------------------------------------------------------: | ------: | --------------------------------------------------------------- |
| `score_type`          | `"lin"`, `"gam"`, `"spline"`, `"krr"`, `"gp"`, `"ff"`     | `"lin"` | Base scoring or regression model used for local causal scores. |
| `mix_type`            | `"lin"`, `"quadratic"`, `"cubic"`, `"nspline"`, `"bspline"` | `"lin"` | Mixture-regression model family used by the CMM local score.   |
| `k_max`               | positive `int`                                            |     `5` | Maximum number of mixture components considered.                |
| `lambda_mix`          | non-negative `float`                                      |   `1.0` | Regularization strength for mixture scoring.                    |
| `hybrid_mixing`       | `bool`                                                    |  `True` | Whether to use hybrid mixture scoring.                          |
| `postprocessing_mode` | `"skip"`, `"edge-strengths"`                              | `"skip"` | Optional postprocessing after graph discovery.                  |
| `score_kwargs`        | `dict[str, Any] \| None`                                  |  `None` | Additional keyword arguments passed to the scoring method.      |
