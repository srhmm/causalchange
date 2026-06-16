
### Algorithms

---

### Tabular causal discovery with TOPIC [[1]](references.md)

**Notebook** [`01_topic_tutorial.ipynb`](../../notebooks/01_topic_tutorial.ipynb)



**Example usage**

```python
import pandas as pd
from causalchange import Topic

X = pd.DataFrame(...)

cc = Topic(score_type="lin")

cc.fit(X)
print(cc.graph_.edges())
```


---

### Multi-context tabular data with LINC  [[2]](references.md)

**Notebook** [`02_linc_tutorial.ipynb`](../../notebooks/02_linc_tutorial.ipynb)

**Example usage**

```python
import pandas as pd
from causalchange import Linc

X = pd.DataFrame(...)
cc = Linc(
    score_type="lin",
    context_col="context",
)
cc.fit(X)
```


---

### Time series causal discovery with SpaceTime  [[3]](references.md)

**Notebook** [`03_spacetime_tutorial.ipynb`](../../notebooks/03_spacetime_tutorial.ipynb)

**Example usage**

```python
import pandas as pd
from causalchange import SpaceTime

X = pd.DataFrame(...)
cc = SpaceTime(
    data_mode="time-contexts",
    score_type="lin",
    context_col="context",
    tau_max=2,
    changepoint_mode="detect",
    changepoint_scope="global",
    changepoint_method="pelt",
    clustering_scope="regimes-contexts",
    clustering_method="statistical-testing",
    testing_method="kernel",
)
cc.fit(X)
```

For `changepoint_mode="detect"`, install `ruptures`.
For kernel mechanism testing with `testing_method="kernel"`, install `hyppo`.
Both are included in `pip install "causalchange[spacetime]"`.

---

### Causal Clustering with CMMs  [[4]](references.md)
**Notebook** [`04_cmm_tutorial.ipynb`](../../notebooks/04_cmm_tutorial.ipynb)

**Example usage**
```python
import pandas as pd
from causalchange import CMM

X = pd.DataFrame(...)
model = CMM(
    score_type="lin",
    mix_type="lin",
    k_max=2,
    lambda_mix=0.0,
    hybrid_mixing=False,
    seed=0,
)

model.fit(X)
```

---
See also the [Inputs](inputs.md) and [Outputs](outputs.md) documentation.
