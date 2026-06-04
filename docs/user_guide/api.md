### API reference

---




#### TOPIC
use `Topic` for one tabular dataset


```python
cc = Topic(score_type="lin")
cc.fit(X)

print(cc.graph_.edges())
```

---


#### LINC

use `Linc` for  multi-context tabular datasets, where `X` contains a context column identifying the dataset/context for each row.

```python
cc = Linc(
    score_type="lin",
    context_col="context",
)

cc.fit(X)
print(cc.graph_.edges())
```

---

#### SpaceTime

use `SpaceTime` for time series
```python
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

print(cc.graph_.edges())
print(cc.changepoints_)
```

For advanced/custom configurations, use the [dev API](../dev_guide/architecture.md).
See also the [Inputs](inputs.md) and [Outputs](outputs.md) documentation.
