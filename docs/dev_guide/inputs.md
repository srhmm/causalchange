### Inputs

---

### Input Data

`causalchange` supports the following input data

| Mode                     | Input |
|--------------------------|---|
| `DataMode.TABULAR`       | one tabular dataset |
| `DataMode.TAB_CONTEXTS`  | tabular data from multiple contexts |
| `DataMode.TIME`          | one time series |
| `DataMode.TIME_CONTEXTS` | time series from multiple contexts |

Note: For context data, `context_col` gives the column containing context labels.
For temporal data, `tau_max` gives the maximum time lag/window size to construct nodes of the form `("x", lag)`.


### Input Parameters


| Concept | Enum | Common values |
|---|---|---|
| Input data | `DataMode` | `TABULAR`, `TAB_CONTEXTS`, `TIME`, `TIME_CONTEXTS` |
| Graph search | `GraphSearch` | `TOPIC` for tabular, `GLOBE` for temporal |
| Tabular score | `ScoreType` | `LIN`, `GAM`, `SPLINE`, `KRR` |
| GP / RFF score | `GPType` | `EXACT`, `FOURIER` |
| Observed tabular contexts | `TabularContextMode` | `SKIP`, `ORACLE`, `DETECT` |
| Context combination | `TabularContextMethod` | `SKIP`, `CHAIN`, `LINC` |
| Changepoints | `ChangepointMode` | `SKIP`, `ORACLE`, `DETECT` |
| Changepoint scope | `ChangepointScope` | `GLOBAL`, `PER_CONTEXT` |
| Changepoint method | `ChangepointMethod` | `PELT` |
| Mechanism partition scope | `MechanismClusteringScope` | `SKIP`, `REGIMES`, `CONTEXTS`, `REGIMES_CONTEXTS` |
| Mechanism partition method | `MechanismClusteringMethod` | `TESTING` currently; `CLUSTERING` is declared but should be treated as not implemented unless you add it |
| Mechanism equality test | `StatisticalTestingMethod` | `KERNEL`, `NONE` |
| Postprocessing | `PostprocessingMode` | `SKIP`, `EDGE_STRENGTHS` |
