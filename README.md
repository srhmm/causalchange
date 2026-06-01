## CausalChange

Implementations of causal discovery algorithms for settings in which causal mechanisms may change across contexts, regimes, or time.

---

### Setup

 Install the core package with `pip install causalchange`. For SpaceTime changepoint detection and mechanism testing, use
 `pip install causalchange[spacetime]`.

**Dev Install** To use with conda and install from source,
```bash
git clone https://github.com/srhmm/causalchange.git
cd causalchange
conda create -n causalchange python=3.10 -y
conda activate causalchange
pip install -e ".[dev,spacetime,notebooks]"
```

**Notebooks** To run the notebooks, use `pip install "causalchange[spacetime]` or

```bash
pip install ipykernel
python -m ipykernel install --user --name causalchange --display-name "Python (cc)"
```

---

### Quick Start

The demos under `notebooks/` show basic usage on small synthetic examples,
*   [TOPIC](notebooks/02_topic_tutorial.ipynb), for score-based causal DAG discovery from tabular data in topological order [2],
*   [LINC](notebooks/03_linc_tutorial.ipynb), for causal discovery from multiple contexts, i.e., multiple tabular datasets with distribution shifts [1],
*   [SpaceTime](notebooks/04_spacetime_tutorial.ipynb), for temporal causal discovery and changepoint detection in time series or multi-context time series [3].


---

### Algorithms

#### Tabular causal discovery with TOPIC

TOPIC [2] is a score-based causal discovery method for tabular data that searches over topological orders.
In this library it is implemented as a modular graph search backend and can be combined with different local MDL scores.

```python
import pandas as pd

from causalchange.causal_change import CausalChange
from causalchange.config.cc_types import (
    ContextAggregation,
    DataMode,
    GraphSearch,
    ScoreType,
)

X = pd.DataFrame(...)

cc = CausalChange(
    data_mode=DataMode.IID,
    graph_search=GraphSearch.TOPIC,
    score_type=ScoreType.LIN,
    aggregation=ContextAggregation.SKIP,
)

cc.fit(X)

print(cc.graph_.edges())
```

#### Multi-context tabular data with LINC

For tabular data from multiple contexts, the library supports an extension of TOPIC to multiple contexts
using the ideas described in the LINC paper [1].
For tabular data with multiple contexts, the context column is specified through `context_col`.

```python
cc = CausalChange(
    data_mode=DataMode.CONTEXTS,
    graph_search=GraphSearch.TOPIC,
    score_type=ScoreType.LIN,
    aggregation=ContextAggregation.LINC,
    context_col="context",
)

cc.fit(X)
```

Temporal multi-context data is handled separately through SpaceTime.

#### Time series causal discovery with SpaceTime

SpaceTime is a score-based causal discovery method for time series, optionally with multiple contexts and changepoints.
The current implementation supports both single time series as well as multi-context time series.
For this, use `DataMode.TIME` or `DataMode.TIME_CONTEXTS`.

```python
from causalchange.config.cc_config import ChangepointMode

cc = CausalChange(
    data_mode=DataMode.TIME_CONTEXTS,
    graph_search=GraphSearch.GLOBE,
    score_type=ScoreType.LIN,
    aggregation=ContextAggregation.SKIP,
    context_col="context",
    tau_max=2,
    changepoints=ChangepointMode.DETECT,
    d_min=20,
    pelt_penalty=1.0,
    detect_contexts=True,
    detect_regimes=True,
)

cc.fit(X)
```

SpaceTime uses temporal nodes of the form

```python
("x0", 0)  # current time
("x0", 1)  # lag 1
("x0", 2)  # lag 2
```

and learns directed edges into lag-0 variables. For example, `(("x0", 1), ("x1", 0))` means `x0(t-1) -> x1(t)`.
For `ChangepointMode.DETECT` and mechanism partitioning with `detect_contexts=True` or `detect_regimes=True`, install the SpaceTime extra with `pip install "causalchange[spacetime]"`.

---

### Tests

Run the full test suite with

```bash
pytest
```

Run linting and formatting with

```bash
ruff check .
ruff format .
```

Before committing, the repository uses pre-commit hooks.

```bash
pre-commit run --all-files
```

---

### References

[1] Mameche, S., Kaltenpoth, D., and Vreeken, J. *Learning Causal Models under Independent Changes.* NeurIPS, 2023.

[2] Xu, S., Mameche, S., and Vreeken, J. *Information-theoretic Causal Discovery in Topological Order.* AISTATS, 2025.

[3] Mameche, S., Cornanguer, L., Ninad, U., and Vreeken, J. *SpaceTime: Causal Discovery from Non-stationary Time Series.* AAAI, 2025.

[4] Mameche, S., Kalofolias, J., and Vreeken, J. *Causal Mixture Models: Characterization and Discovery.* NeurIPS, 2025.
