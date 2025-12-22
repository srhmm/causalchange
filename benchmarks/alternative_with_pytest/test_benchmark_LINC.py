from __future__ import annotations

import time

import pytest
import networkx as nx

from pgmpy.causal_discovery.LINC import LINC
from pgmpy.benchmarks.synthetic.metrics import compute_metrics
from pgmpy.benchmarks.synthetic.generators import (
    sample_multicontext_linear_gaussian_interventional,
    sample_multicontext_nonlinear_additive_interventional,
)


@pytest.mark.bench
@pytest.mark.parametrize("scoring_method", ["aic-g", "bic-g"])
def test_bench_linc_fit_smoke(scoring_method, benchrec):
    df, _, _, _, _ = sample_multicontext_linear_gaussian_interventional(
        n_samples_per_context=100,
        n_nodes=5,
        edge_prob=0.4,
        n_contexts=3,
        seed=42,
        context_col="context",
        n_intervened_per_context=1,
    )

    est = LINC(context_col="context", scoring_method=scoring_method)

    t0 = time.perf_counter()
    dag = est.fit(df)
    t1 = time.perf_counter()

    assert dag is not None
    benchrec.add_series(bench="linc_fit", scoring_method=scoring_method, metric="time_s", values=[t1 - t0])


@pytest.mark.bench
@pytest.mark.metrics
@pytest.mark.parametrize("scoring_method", ["aic-g", "bic-g"])
@pytest.mark.parametrize("generator", ["linear_iv", "nonlinear_iv"])
@pytest.mark.parametrize("intervention", ["hard", "soft_weight", # "soft_mechanism",
                                          "shift", "noise"])
def test_bench_linc_fit_with_metrics(scoring_method, generator, intervention, benchrec):
    n_nodes = 5
    edge_prob = 0.4
    n_contexts = 5
    n_samples_per_context = 500
    n_repeats = 20
    seed = 42

    shds, edge_f1s, skel_f1s, times = [], [], [], []

    for r in range(n_repeats):
        if generator == "linear_iv":
            df, _, true_inv, _, _ = sample_multicontext_linear_gaussian_interventional(
                n_samples_per_context=n_samples_per_context,
                n_nodes=n_nodes,
                edge_prob=edge_prob,
                n_contexts=n_contexts,
                seed=seed + r,
                context_col="context",
                intervention_type=intervention,
                n_intervened_per_context=2,
            )
        else:
            df, _, true_inv, _, _ = sample_multicontext_nonlinear_additive_interventional(
                n_samples_per_context=n_samples_per_context,
                n_nodes=n_nodes,
                edge_prob=edge_prob,
                n_contexts=n_contexts,
                seed=seed + r,
                context_col="context",
                intervention_type=intervention,
                n_intervened_per_context=2,
                nonlinearity="tanh",
            )

        est = LINC(context_col="context", scoring_method=scoring_method)

        t0 = time.perf_counter()
        dag = est.fit(df)
        t1 = time.perf_counter()

        est_nx = nx.DiGraph()
        est_nx.add_nodes_from([str(n) for n in dag.nodes()])
        est_nx.add_edges_from([(str(u), str(v)) for (u, v) in dag.edges()])

        m = compute_metrics(true_inv, est_nx)
        shds.append(float(m.shd))
        edge_f1s.append(float(m.edge_f1))
        skel_f1s.append(float(m.skel_f1))
        times.append(t1 - t0)

    bench_name = f"linc_fit-{generator}_iv-{intervention}"
    #benchrec.add_series(bench=bench_name, scoring_method=scoring_method, metric="shd", values=shds)
    benchrec.add_series(bench=bench_name, scoring_method=scoring_method, metric="edge_f1", values=edge_f1s)
    benchrec.add_series(bench=bench_name, scoring_method=scoring_method, metric="skel_f1", values=skel_f1s)
