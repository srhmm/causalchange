from __future__ import annotations
import os
import pytest

from exp.exp_change.gen.generate import FunType, IvType, NoiseType
from causalchange.old.causal_change_large import CausalChange
from causalchange._cc_types import DataMode, GraphSearch, ScoreType
from causalchange.old.scoring.fit_cond_mixture import MixingType

from benchmarks.synthetic.generators import (
    SynthSpec,
    sample_iid,
    sample_contexts,
    sample_mixed,
    sample_time_series,
    sample_time_series_contexts,
)
from benchmarks.metrics import ( compute_metrics,
    n_directed_edges,
    n_skeleton_edges,
    normalized_shd, )


def _seeds() -> list[int]:
    # override with BENCH_SEEDS=0,1,2,3 or BENCH_NSEEDS=20
    if "BENCH_SEEDS" in os.environ:
        return [int(x) for x in os.environ["BENCH_SEEDS"].split(",") if x.strip() != ""]
    n = int(os.environ.get("BENCH_NSEEDS", "10"))
    base = int(os.environ.get("BENCH_SEED0", "0"))
    return list(range(base, base + n))


@pytest.mark.bench
@pytest.mark.parametrize(
    "data_mode",
    [DataMode.IID, DataMode.CONTEXTS, DataMode.MIXED, DataMode.TIME, DataMode.TIME_CONTEXTS],
)
@pytest.mark.parametrize(
    "graph_search",
    [GraphSearch.TOPIC, GraphSearch.GLOBE],
)
@pytest.mark.parametrize(
    "score_type",
    [ScoreType.SPLINE] #, ScoreType.KRR, ScoreType.GAM, GPType.FOURIER, ScoreType.LIN],
)
@pytest.mark.parametrize(
    "fun_type",
    [FunType.MIX, FunType.LIN]
)

@pytest.mark.parametrize(
    "iv_type",
    [IvType.COEF, IvType.SHIFT]
)
def test_bench_end_to_end(data_mode, graph_search, score_type, fun_type, iv_type, benchrec):
    if not graph_search.is_compatible_with(data_mode):
        pytest.skip(f"{graph_search} not compatible with data_mode {data_mode}")

    seeds = _seeds()

    N = int(os.environ.get("BENCH_NNODES", "5"))
    #DEG = float(os.environ.get("BENCH_DEG", "2.0"))
    #P = float(os.environ.get("BENCH_P", str(min(0.9, DEG / max(1, (N - 1))))))
    P = float(os.environ.get("BENCH_P", "0.4"))
    D = int(os.environ.get("BENCH_NSAMPLES", "500"))

    spec = SynthSpec(
        n_nodes=N,
        edge_prob=P,
        fun_type=fun_type,
        iv_type=iv_type,
        noise_type=NoiseType.GAUSS,
        pc=0.5,
        kmn=1,
        kmx=2,
    )


    edge_f1s, skel_f1s, shds = [], [], []
    n_edges_true, n_edges_hat,n_skel_true, n_skel_hat, edge_ps, edge_rs, skel_ps, skel_rs, nshds = [], [], [], [], [], [], [], [], []

    for s in seeds:
        if data_mode == DataMode.IID:
            X, true_g = sample_iid(spec, n_samples=D, seed=s)

            cc = CausalChange(
                data_mode=DataMode.IID,
                graph_search=graph_search,
                score_type=score_type,
                truths={"true_g": true_g},
                vb=0,
            )
            est_g = cc.fit(X)

        elif data_mode == DataMode.CONTEXTS:
            X_ctxs, true_g = sample_contexts(
                spec,
                n_contexts=3,
                n_samples_per_context=max(200, D // 3),
                seed=s,
            )
            cc = CausalChange(
                data_mode=DataMode.CONTEXTS,
                graph_search=graph_search,
                score_type=score_type,
                truths={"true_g": true_g},
                vb=0,
            )
            est_g = cc.fit(X_ctxs)

        elif data_mode == DataMode.MIXED:
            pytest.importorskip("rpy2", reason="MIXED mode requires rpy2")

            X, true_g = sample_mixed(spec, n_samples=D, seed=s)
            cc = CausalChange(
                data_mode=DataMode.MIXED,
                graph_search=graph_search,
                mixing_type=MixingType.MIX_LIN,
                truths={"true_g": true_g},
                vb=0,
            )
            est_g = cc.fit(X)

        elif data_mode == DataMode.TIME:
            pytest.xfail("TIME synthetic generation not implemented (src.causalchange.gen)")
            X, true_g = sample_time_series(spec, n_timepoints=D, seed=s)
            cc = CausalChange(
                data_mode=DataMode.TIME,
                graph_search=graph_search,
                score_type=score_type,
                truths={"true_g": true_g},
                vb=0,
            )
            try:
                est_g = cc.fit(X)
            except NotImplementedError:
                pytest.xfail("TIME graph search not implemented")
                return

        elif data_mode == DataMode.TIME_CONTEXTS:
            pytest.xfail("TIME_CONTEXTS synthetic generation not implemented (src.causalchange.gen)")
            X_ctxs, true_g = sample_time_series_contexts(spec, n_contexts=3, n_timepoints_per_context=max(200, D // 3), seed=s)
            cc = CausalChange(
                data_mode=DataMode.TIME_CONTEXTS,
                graph_search=graph_search,
                score_type=score_type,
                truths={"true_g": true_g},
                vb=0,
            )
            try:
                est_g = cc.fit(X_ctxs)
            except NotImplementedError:
                pytest.xfail("TIME_CONTEXTS graph search not implemented")
                return
        else:
            raise RuntimeError(f"Unhandled data_mode: {data_mode}")

        m = compute_metrics(true_g, est_g)
        edge_f1s.append(m.edge_f1)
        skel_f1s.append(m.skel_f1)
        shds.append(float(m.shd))

        n_edges_true.append(n_directed_edges(true_g))
        n_edges_hat.append(n_directed_edges(est_g))

        n_skel_true.append(n_skeleton_edges(true_g))
        n_skel_hat.append(n_skeleton_edges(est_g))

        edge_ps.append(m.edge_precision)
        edge_rs.append(m.edge_recall)
        skel_ps.append(m.skel_precision)
        skel_rs.append(m.skel_recall)

        nshds.append(normalized_shd(true_g, est_g))



    for (nm, record) in [
            ("edge_f1", edge_f1s),
            ("skel_f1", skel_f1s),
            ("shd", shds),
            ("shd_norm", shds),
            ("edge_p", edge_ps),
            ("edge_r", nshds),
            ("n_edges_true", n_edges_true),
            ("n_edges_hat", n_edges_hat),
            ("skel_p", skel_ps),
            ("skel_h", skel_rs),
            ("n_skel_true", n_skel_true),
            ("n_skel_hat", n_skel_hat),
        ]:
        benchrec.add_series("end_to_end", str(data_mode), str(graph_search), str(score_type), str(fun_type), str(iv_type), nm, record)

