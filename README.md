## CausalChange

causalchange provides causal discovery algorithms with focus on addressing different forms of
distribution shifts.

---

### Setup

 causalchange is available on PyPi. Install the core package with `pip install causalchange`, with additional
 dependencies needed for the SpaceTime algorithm using `pip install causalchange[spacetime]`, resp. notebooks using
 `pip install causalchange[notebooks]`.

---

### Quick Example

```python
import pandas as pd

from causalchange import Topic

X = pd.DataFrame(...)


cc = Topic(score_type="lin")
cc.fit(X)
print(cc.graph_.edges())
```

---

### Further Examples

The `notebooks/` show basic usage on small synthetic examples,
*   [TOPIC](notebooks/01_topic_tutorial.ipynb), for score-based causal DAG discovery from tabular data in topological order [2],
*   [LINC](notebooks/02_linc_tutorial.ipynb), for causal discovery from multiple contexts, i.e., multiple tabular datasets with distribution shifts [1],
*   [SpaceTime](notebooks/03_spacetime_tutorial.ipynb), for temporal causal discovery and changepoint detection in time series or multi-context time series [3].

---
### Documentation

See [docs/](docs/) for additional documentation.


---

>**References**
>
>[1] Mameche, S., Kaltenpoth, D., and Vreeken, J. *Learning Causal Models under Independent Changes.* NeurIPS, 2023.
>
> [2] Xu, S., Mameche, S., and Vreeken, J. *Information-theoretic Causal Discovery in Topological Order.* AISTATS, 2025.
>
> [3] Mameche, S., Cornanguer, L., Ninad, U., and Vreeken, J. *SpaceTime: Causal Discovery from Non-stationary Time Series.* AAAI, 2025.
>
> [4] Mameche, S., Kalofolias, J., and Vreeken, J. *Causal Mixture Models: Characterization and Discovery.* NeurIPS, 2025.
