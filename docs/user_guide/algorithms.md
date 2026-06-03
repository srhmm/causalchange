
### Algorithms

---

#### Tabular causal discovery with TOPIC [[1]](references.md)

Notebook [`02_topic_tutorial.ipynb`](../../notebooks/02_topic_tutorial.ipynb)

Setting
```python
data_mode=DataMode.IID
graph_search=GraphSearch.TOPIC
````
Example usage
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


---

#### Multi-context tabular data with LINC  [[2]](references.md)

Notebook [`03_linc_tutorial.ipynb`](../../notebooks/03_linc_tutorial.ipynb)

Setting
```python
data_mode=DataMode.CONTEXTS
context_mode=ContextMode.LINC
```
Example usage
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


---

#### Time series causal discovery with SpaceTime  [[3]](references.md)

Notebook [`04_spacetime_tutorial.ipynb`](../../notebooks/04_spacetime_tutorial.ipynb)

 Settings
```python
data_mode=DataMode.TIME
#or
data_mode=DataMode.TIME_CONTEXTS
```

Example Usage

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

#### Causal Clustering with CMMs  [[4]](references.md)
Under construction.
