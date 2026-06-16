### Quickstart

---

```python
import numpy as np
import pandas as pd

from causalchange import Topic

rng = np.random.default_rng(0)
n = 300

# chain: X -> Y -> Z
x = rng.normal(size=n)
y = 2.0 * x + 0.2 * rng.normal(size=n)
z = -1.0 * y + 0.2 * rng.normal(size=n)

X = pd.DataFrame({"X": x, "Y": y, "Z": z})

cc = Topic(score_type="lin", seed=0)
cc.fit(X)

print("Topological order:", cc.topological_order_)
print("Edges:", sorted(cc.graph_.edges()))
```

See [Algorithms](../user_guide/algorithms.md) for additional usage examples.
