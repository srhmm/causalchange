## CausalChange

 `causalchange` provides implementations of some score-based algorithms for causal discovery,
 with focus on addressing different forms of distribution shifts.

---

### Setup

Install the core package with `pip install causalchange`, and with additional
dependencies for the resp. algorithms using `pip install causalchange[spacetime]` and `pip install causalchange[cmm]`.

---

### Quick Example

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

---

### Further Examples

See the `notebooks/` for basic usage on small synthetic examples. The algorithms are
*   [TOPIC](notebooks/01_topic_tutorial.ipynb), for score-based causal DAG discovery from tabular data in topological order [1],
*   [LINC](notebooks/02_linc_tutorial.ipynb), for score-based causal discovery from multiple contexts, i.e., multiple tabular datasets under latent distribution shifts/interventions [2],
*   [CMM](notebooks/04_cmm_tutorial.ipynb), for score-based causal discovery from latent mixtures, i.e., a tabular dataset comprised of different hidden populations/contexts/interventions [3].
*   [SpaceTime](notebooks/03_spacetime_tutorial.ipynb), for temporal causal discovery, changepoint detection, and causal clustering analysis in time series or multi-context time series [4].

---
### Documentation

See the [docs/](docs/) for additional documentation.


---

>**References**
>
> [1] Xu, S., Mameche, S., and Vreeken, J. *Information-theoretic Causal Discovery in Topological Order.* AISTATS, 2025.
>
> [2] Mameche, S., Kaltenpoth, D., and Vreeken, J. *Learning Causal Models under Independent Changes.* NeurIPS, 2023.
>
> [3] Mameche, S., Kalofolias, J., and Vreeken, J. *Causal Mixture Models: Characterization and Discovery.* NeurIPS, 2025.
>
> [4] Mameche, S., Cornanguer, L., Ninad, U., and Vreeken, J. *SpaceTime: Causal Discovery from Non-stationary Time Series.* AAAI, 2025.
