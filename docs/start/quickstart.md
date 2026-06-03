### Quickstart

---


```python
import pandas as pd

from causalchange.causal_change import CausalChange
from causalchange.core.types import DataMode, GraphSearch, ScoreType

X = pd.DataFrame(...)

cc = CausalChange(
    data_mode=DataMode.IID,
    graph_search=GraphSearch.TOPIC,
    score_type=ScoreType.LIN,
)

cc.fit(X)

print(cc.graph_.edges())
```

See the tutorials of the [Algorithms](../user_guide/algorithms.md) for larger examples.
