### Data modes

---

`causalchange` supports the following input data

| Mode | Input |
|---|---|
| `DataMode.IID` | one tabular dataset |
| `DataMode.CONTEXTS` | tabular data from multiple contexts |
| `DataMode.TIME` | one time series |
| `DataMode.TIME_CONTEXTS` | time series from multiple contexts |

For context data, `context_col` gives the column containing context labels.

For temporal data, nodes have the form `("x", lag)`.
