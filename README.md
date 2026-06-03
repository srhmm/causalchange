## CausalChange

causalchange provides causal discovery algorithms for various settings, with focus on addressing different forms of
distribution shifts, here modeled as changes in structural causal mechanisms under a causal model. 
Examples include multi-context tabular data across heterogenous populations, or time series with regime shifts over time.

---

### Setup

 causalchange is available on PyPi. Install the core package with `pip install causalchange`, with additional 
 dependencies needed for the SpaceTime algorithm using `pip install causalchange[spacetime]`, resp. notebooks using 
 `pip install causalchange[notebooks]`.

**Dev Install** To use with conda and install from source,
```bash
git clone https://github.com/srhmm/causalchange.git
cd causalchange
conda create -n causalchange python=3.10 -y
conda activate causalchange
pip install -e ".[dev,spacetime,notebooks]"
```

**Notebooks** To run the notebooks,

```bash
pip install ipykernel
python -m ipykernel install --user --name causalchange --display-name "Python (cc)"
```

---

### Quickstart

The demos under `notebooks/` show basic usage on small synthetic examples,
*   [TOPIC](notebooks/02_topic_tutorial.ipynb), for score-based causal DAG discovery from tabular data in topological order [2],
*   [LINC](notebooks/03_linc_tutorial.ipynb), for causal discovery from multiple contexts, i.e., multiple tabular datasets with distribution shifts [1],
*   [SpaceTime](notebooks/04_spacetime_tutorial.ipynb), for temporal causal discovery and changepoint detection in time series or multi-context time series [3].


---

### Examples

#### Tabular causal discovery with TOPIC [2]

```python
import pandas as pd

from causalchange.causal_change import CausalChange
from causalchange.config.cc_types import (
    DataMode,
    GraphSearch,
    ScoreType,
)

X = pd.DataFrame(...)

cc = CausalChange(
    data_mode=DataMode.IID,
    graph_search=GraphSearch.TOPIC,
    score_type=ScoreType.LIN,
)

cc.fit(X)

print(cc.graph_.edges())
```

#### Multi-context tabular data with LINC [1]

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
Above, X contains a column `context_col` encoding the context. 

#### Time series causal discovery with SpaceTime [3]


```python
from causalchange.config.cc_config import ChangepointMode

cc = CausalChange(
    data_mode=DataMode.TIME_CONTEXTS,
    graph_search=GraphSearch.GLOBE,
    score_type=ScoreType.LIN,
    context_col="context",
    tau_max=2,
    changepoints=ChangepointMode.DETECT,
    d_min=20,
    pelt_penalty="auto",
    detect_contexts=True,
    detect_regimes=True,
)

cc.fit(X)
```

SpaceTime uses temporal nodes of the form `("x0", 0)` (current time), `("x0", 1)` (lag 1), `("x0", 2)` (lag 2) and 
learns directed edges of the form `(("x0", 1), ("x1", 0))` meaning `x0(t-1) -> x1(t)`.
For `ChangepointMode.DETECT` and mechanism partitioning with `detect_contexts=True` or `detect_regimes=True`,
install the SpaceTime extra with `pip install "causalchange[spacetime]"`. 
---

### Notes

Run the full test suite with `pytest`.

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
