from __future__ import annotations

import numpy as np
import pandas as pd

from causalchange import SpaceTime, Topic


def smoke_topic() -> None:
    rng = np.random.default_rng(42)
    n = 100

    x0 = rng.normal(size=n)
    x1 = np.tanh(x0) + rng.normal(scale=0.2, size=n)

    X = pd.DataFrame({"x0": x0, "x1": x1})

    cc = Topic(score_type="lin").fit(X)

    assert cc.graph_ is not None
    assert cc.graph_.number_of_nodes() == 2

    print("TOPIC smoke ok")
    print("  edges:", sorted(cc.graph_.edges()))


def smoke_spacetime() -> None:
    n = 80
    cp = 40
    rng = np.random.default_rng(42)

    x0 = np.zeros(n)
    x1 = np.zeros(n)

    for t in range(1, n):
        x0[t] = 0.6 * x0[t - 1] + rng.normal(scale=0.3)

        if t < cp:
            x1[t] = 0.5 * x1[t - 1] + 0.8 * x0[t - 1] + rng.normal(scale=0.3)
        else:
            x1[t] = 0.5 * x1[t - 1] - 0.8 * x0[t - 1] + rng.normal(scale=0.3)

    X = pd.DataFrame({"x0": x0, "x1": x1})

    cc = SpaceTime(
        data_mode="time",
        score_type="lin",
        tau_max=1,
        changepoint_mode="fixed",
        fixed_changepoints=[cp],
        clustering_scope="skip",
    ).fit(X)

    assert cc.graph_ is not None
    assert cc.changepoints_ == [cp]

    print("SpaceTime smoke ok")
    print("  changepoints:", cc.changepoints_)
    print("  edges:", sorted(cc.graph_.edges()))


if __name__ == "__main__":
    smoke_topic()
    smoke_spacetime()
