### Quickstart

---
Topic
```python
import pandas as pd

from causalchange import Topic, ScoreType

X = pd.DataFrame(...)

cc = Topic(score_type="lin")

cc.fit(X)
print(cc.graph_.edges())
```

See the [Algorithm usage examples](../user_guide/algorithms.md) for additional examples.
