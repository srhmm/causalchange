from __future__ import annotations

import numpy as np
import pandas as pd

from causalchange import Linc, SpaceTime, Topic


def make_chain_data(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 2.0 * x + 0.2 * rng.normal(size=n)
    z = -1.0 * y + 0.2 * rng.normal(size=n)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def make_context_chain_data(n_per_context: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []

    for context, shift, slope in [(0, 0.0, 2.0), (1, 1.0, 1.5)]:
        x = rng.normal(loc=shift, size=n_per_context)
        y = slope * x + 0.2 * rng.normal(size=n_per_context)
        z = -1.0 * y + 0.2 * rng.normal(size=n_per_context)
        part = pd.DataFrame({"X": x, "Y": y, "Z": z, "context": context})
        parts.append(part)

    return pd.concat(parts, ignore_index=True)


def make_time_series_data(n: int = 180, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)

    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.normal(scale=0.5)
        y[t] = 1.2 * x[t - 1] + 0.3 * y[t - 1] + rng.normal(scale=0.3)
        z[t] = -0.8 * y[t - 1] + 0.2 * z[t - 1] + rng.normal(scale=0.3)

    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def test_topic_fit_tiny_chain_smoke():
    cc = Topic(score_type="lin", seed=0)
    cc.fit(make_chain_data())

    assert cc.fitted_graph
    assert set(cc.graph_.nodes) == {"X", "Y", "Z"}
    assert cc.topological_order_ is not None
    assert sorted(cc.topological_order_) == ["X", "Y", "Z"]
    assert cc.history_ is not None


def test_linc_fit_tiny_context_chain():
    cc = Linc(score_type="lin", context_col="context", seed=0)
    cc.fit(make_context_chain_data())

    assert cc.fitted_graph
    assert set(cc.graph_.nodes) == {"X", "Y", "Z"}
    assert "context" not in cc.graph_.nodes
    assert cc.topological_order_ is not None


def test_spacetime_fit_tiny_time_series_without_optional_extras():
    cc = SpaceTime(
        score_type="lin",
        data_mode="time",
        tau_max=1,
        changepoint_mode="skip",
        d_min=20,
        max_iter=1,
        seed=0,
    )
    cc.fit(make_time_series_data())

    assert cc.fitted_graph
    assert cc.changepoints_ == []
    assert ("X", 1) in cc.graph_.nodes
    assert ("Y", 0) in cc.graph_.nodes
    assert cc.topological_order_ is not None