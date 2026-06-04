
### Algorithms

---

### Tabular causal discovery with TOPIC [[1]](references.md)

**Notebook** [`02_topic_tutorial.ipynb`](../../notebooks/01_topic_tutorial.ipynb)



**Example usage**

```python
import pandas as pd

from causalchange import Topic, ScoreType

X = pd.DataFrame(...)

cc = Topic(score_type=ScoreType.LIN)

cc.fit(X)
print(cc.graph_.edges())
```


---

### Multi-context tabular data with LINC  [[2]](references.md)

**Notebook** [`03_linc_tutorial.ipynb`](../../notebooks/02_linc_tutorial.ipynb)

**Example usage**

```python
from causalchange import Linc, ScoreType

cc = Linc(
    score_type=ScoreType.LIN,
    context_col="context",
)

cc.fit(X)
```


---

### Time series causal discovery with SpaceTime  [[3]](references.md)

**Notebook** [`04_spacetime_tutorial.ipynb`](../../notebooks/03_spacetime_tutorial.ipynb)

**Example usage**

```python
from causalchange import (
    SpaceTime,
    DataMode,
    ScoreType,
    ChangepointMode,
    ChangepointScope,
    ChangepointMethod,
    MechanismClusteringScope,
    MechanismClusteringMethod,
    StatisticalTestingMethod,
)

cc = SpaceTime(
    data_mode=DataMode.TIME_CONTEXTS,
    score_type=ScoreType.LIN,
    context_col="context",
    tau_max=2,
    changepoint_mode=ChangepointMode.DETECT,
    changepoint_scope=ChangepointScope.GLOBAL,
    changepoint_method=ChangepointMethod.PELT,
    clustering_scope=MechanismClusteringScope.REGIMES_CONTEXTS,
    clustering_method=MechanismClusteringMethod.TESTING,
    testing_method=StatisticalTestingMethod.KERNEL,
)

cc.fit(X)
```

SpaceTime uses temporal nodes of the form `("x0", 0)` (current time), `("x0", 1)` (lag 1), `("x0", 2)` (lag 2) and
learns directed edges of the form `(("x0", 1), ("x1", 0))` meaning `x0(t-1) -> x1(t)`.
For `ChangepointMode.DETECT`, install `ruptures` via `pip install "causalchange[spacetime]"`. For kernel mechanism testing with `testing_method=StatisticalTestingMethod.KERNEL`, install the full SpaceTime extras, which include `hyppo`.

---

#### Causal Clustering with CMMs  [[4]](references.md)
Under construction.
