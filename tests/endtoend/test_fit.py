from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalchange import CMM, Linc, SpaceTime, Topic
from causalchange.core.types import MixedSCMType
from causalchange.scoring.regression import fit_regression_mixture
from tests.util import has_rpy2_flexmix


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
        changepoint_scope="skip",
        changepoint_method="skip",
        clustering_scope="skip",
        clustering_method="skip",
        testing_method="skip",
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


@pytest.mark.skipif(
    not has_rpy2_flexmix(),
    reason="requires R, rpy2, and the R package flexmix",
)
def test_cmm_wrapper_fits_small_dataframe():
    rng = np.random.default_rng(0)
    n = 80

    x0 = rng.normal(size=n)
    z = rng.integers(0, 2, size=n)
    x1 = (1.0 + z) * x0 + rng.normal(scale=0.2, size=n)
    x2 = x1 + rng.normal(scale=0.2, size=n)

    df = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})

    est = CMM(mix_type="lin", k_max=2).fit(df)

    assert est.graph_ is not None
    assert set(est.graph_.nodes()) == {"X0", "X1", "X2"}
    assert est.public_config_.mix_type == MixedSCMType.LIN


@pytest.mark.skipif(
    not has_rpy2_flexmix(),
    reason="requires R, rpy2, and the R package flexmix",
)
def test_cmm_wrapper_exposes_mixture_components():
    rng = np.random.default_rng(0)
    n = 80

    x0 = rng.normal(size=n)
    z = rng.integers(0, 2, size=n)
    x1 = (1.0 + z) * x0 + rng.normal(scale=0.2, size=n)
    x2 = x1 + rng.normal(scale=0.2, size=n)

    df = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})

    est = CMM(
        mix_type="lin",
        k_max=2,
        seed=0,
    ).fit(df)

    assert est.graph_ is not None
    assert est.cmm_components_ is not None
    assert est.cmm_components_ is est.cmm_components_

    mixture = est.cmm_components_

    assert mixture.global_labels is None
    assert mixture.global_responsibilities is None
    assert mixture.target_components

    for target, target_result in mixture.target_components.items():
        assert target_result.target == target
        assert target_result.parents == tuple(sorted(est.graph_.predecessors(target), key=repr))

        assert len(target_result.labels) == len(df)
        assert len(target_result.responsibilities) == len(df)
        assert len(target_result.component_weights) == target_result.n_components

        assert target_result.n_components is not None
        assert target_result.n_components >= 1
        assert target_result.score is not None

        for row in target_result.responsibilities:
            assert len(row) == target_result.n_components
            assert np.isclose(sum(row), 1.0, atol=1e-6)


def test_fit_conditional_mixture_without_parents_returns_labels_and_responsibilities():
    rng = np.random.default_rng(0)
    n = 60

    z = rng.integers(0, 2, size=n)
    x0 = rng.normal(loc=4.0 * z, scale=0.2)
    x1 = 2.0 * x0 + rng.normal(scale=0.1, size=n)
    X = np.column_stack([x0, x1])

    res = fit_regression_mixture(
        MixedSCMType.LIN,
        X=X,
        node_i=0,
        pa_i=[],
        range_k=range(1, 3),
        resid=None,
        true_idl=None,
    )

    assert "idl" in res
    assert "pproba" in res
    assert res["idl"].shape == (n,)
    assert res["pproba"].shape[0] == n
    assert res["best_k"] >= 1


@pytest.mark.skipif(
    not has_rpy2_flexmix(),
    reason="requires R, rpy2, and the R package flexmix",
)
def test_cmm_mixture_result_helpers():
    rng = np.random.default_rng(1)
    n = 80

    x0 = rng.normal(size=n)
    z = rng.integers(0, 2, size=n)
    x1 = (1.0 + z) * x0 + rng.normal(scale=0.2, size=n)
    x2 = x1 + rng.normal(scale=0.2, size=n)

    df = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})

    est = CMM(
        mix_type="lin",
        k_max=2,
        seed=1,
    ).fit(df)

    mixture = est.cmm_components_
    assert mixture is not None

    target = next(iter(mixture.target_components))

    assert mixture.labels_for(target) == mixture.target_components[target].labels
    assert mixture.responsibilities_for(target) == mixture.target_components[target].responsibilities
    assert mixture.parents_for(target) == mixture.target_components[target].parents


def test_topic_wrapper_has_no_mixture_components():
    rng = np.random.default_rng(0)
    n = 60

    x0 = rng.normal(size=n)
    x1 = 2.0 * x0 + rng.normal(scale=0.2, size=n)
    x2 = x1 + rng.normal(scale=0.2, size=n)

    df = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})

    est = Topic(score_type="lin", seed=0).fit(df)

    assert est.graph_ is not None
    assert est.cmm_components_ is None
    assert est.cmm_components_ is None


def test_linc_fit_runs_and_exposes_context_combination_result():
    X = pd.DataFrame(
        {
            "context": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "x": [0.0, 1.0, 2.0] * 3,
            "y": [
                0.0,
                1.0,
                2.0,  # context 0: y = x
                0.0,
                1.0,
                2.0,  # context 1: y = x
                0.0,
                2.0,
                4.0,  # context 2: y = 2x
            ],
        }
    )

    est = Linc(
        score_type="lin",
        context_col="context",
        seed=0,
    ).fit(X)

    assert est.graph_ is not None
    assert "context" not in est.graph_.nodes
    assert est.linc_components_() is not None
    assert "groups" in est.linc_components_().diagnostics
