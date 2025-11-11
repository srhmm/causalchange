from collections import defaultdict
from types import SimpleNamespace

import networkx as nx
import numpy as np

from src.exp.exp_change.algos import CD, OracleType
from src.exp.exp_change.gen.generate import DagType, NoiseType, FunType
from src.exp.exp_change.gen.generate import IvMode, IvType, GSType
from src.exp.exp_change.gen.generate import gen_data_type
from src.exp.exp_change.util.run_util import run_info, run_get_plot_filename
from src.causalchange.mixing.mixing import MixingType, fit_mixture_model
from src.causalchange.causal_change_topo import CausalChangeTopological
from src.causalchange.util.util import compare_Z, lmg_to_directed_edge_adj


#this function should be moved
def run_dgen(options, params, case, exp, rep_seed, sub_seed):
    # Generate Data
    run_info("", options.logger, options.verbosity)
    run_info(f'*** Rep {rep_seed[0] + 1}/{options.reps}, trial {sub_seed} ***', options.logger, options.verbosity)
    run_info(f"*Exp: {', '.join([f'{ky}: {vl}' for ky, vl in exp.items()])}", options.logger, options.verbosity)
    run_info(f"*Params: {case}", options.logger, options.verbosity)
    plot_dir = run_get_plot_filename(options, exp, case, rep_seed[0] + 1)

    seed = rep_seed[1].generate_state(1)[0]
    np.random.seed(seed)

    dataparams = params.copy()
    dataparams.update(
        {"DG": DagType(exp["DG"]), "NS": NoiseType(exp["NS"]), "F": FunType(exp["F"]), "IVM": IvMode(exp["IVM"]),
         "IVT": IvType(exp["IVT"]), "GS": GSType(exp["GS"])})
    X, truths = gen_data_type(dataparams, seed, options.verbosity, options.logger)
    if options.visu and '_dg' in truths: truths['_dg'].plot_X(X, plot_dir)
    run_info("", options.logger, options.verbosity)
    return X, truths


def run_case_mixseparation(options, params, case, exp, rep_seed, sub_seed):
    """Causal cluster label evaluation with random dags"""
    results = defaultdict(SimpleNamespace)
    type_of_oracle_knowledge = options.get_oracles()
    type_of_mixing_algos = options.get_mixing_algos()
    type_of_causal_discovery_algos = options.methods

    X, truths = run_dgen(options, params, case, exp, rep_seed, sub_seed)
    for mixing_ty in type_of_mixing_algos:
        if mixing_ty.is_unconditional_mixture():  # clustering/mixture model: no graph needed
            results = run_case_clustering_method(
                results, X, truths, CD.SKIP, OracleType.SKIP,
                mixing_ty, options, params, case, exp, rep_seed)
        else:  # causal mixture model: discover graph or given graph
            for oracle_ty in type_of_oracle_knowledge:  # decides which graph we use
                if oracle_ty.haveto_discover_G():
                    for method_ty in type_of_causal_discovery_algos:  # different discovery methdos f graph
                        results = run_case_clustering_method(results, X, truths, method_ty, oracle_ty, mixing_ty,
                                                             options, params, case, exp, rep_seed)
                else:
                    results = run_case_clustering_method(results, X, truths, CD.SKIP, oracle_ty, mixing_ty, options,
                                                         params, case, exp, rep_seed)
    return results


def run_case_clustering_method(
        results: dict, X: np.ndarray, truths: dict,
        method_ty: CD, oracle_ty: OracleType, mixing_ty: MixingType, options, params, case, exp, rep_seed):
    SKIP_PRUNING = True  # no causal discovery here
    ORACLE_Z = oracle_ty.is_Z_known()
    ORACLE_K = oracle_ty.is_K_known()
    ORACLE_G = oracle_ty.is_G_known()
    DISCOVER_G = False
    DISCOVER_MIXING_EACH_NODE = mixing_ty.search_each_node()

    plot_dir = run_get_plot_filename(options, exp, case, rep_seed[0] + 1)
    t_A, t_n_Z, t_Z = nx.to_numpy_array(truths['true_g']), truths['t_n_Z'], truths['t_Z']

    if options.verbosity > 0:
        run_info(
            f"\tMETHOD: {mixing_ty} \t({'G*' if oracle_ty.is_G_known() else 'Gempty' if oracle_ty.is_G_empty() else 'Gdense (based on true order)' if oracle_ty.is_G_dense() else f'G discovered using {method_ty.value}'},  {'K*' if ORACLE_K else f'K in [0-{options.KMAX}]'}{', Z*' if ORACLE_Z else ''})",
            options.logger, options.verbosity)

    # Causal Graph under which to fit CMMs
    given_A = None
    if oracle_ty.is_G_known():
        given_A = nx.to_numpy_array(truths['true_g'])
    elif oracle_ty.is_G_empty():
        given_A = np.zeros(t_A.shape) #, create_using=nx.DiGraph)
    elif oracle_ty.is_G_dense():
        given_A = nx.from_numpy_array(np.zeros(t_A.shape), create_using=nx.DiGraph)
        top_order = list(nx.topological_sort(truths['true_g']))
        [given_A.add_edge(nodej, nodei) for ii, nodei in enumerate(top_order) for nodej in
         [nj for ij, nj in enumerate(top_order) if ij < ii]]
        given_A = nx.to_numpy_array(given_A)
    else:
        DISCOVER_G = True
        mth = method_ty.get_method()
        kwargs = dict(mixing_type=mixing_ty, truths=truths, hybrid=False, oracle_Z=ORACLE_Z, oracle_K=ORACLE_K,
                      oracle_G=given_A,
                      lg=options.logger, kmax=options.KMAX)  # todo args for other methods in options
        mth.fit(X, **kwargs)
        given_A = lmg_to_directed_edge_adj(mth.lmg)

    # CMMs: our causal mixture discovery
    # (and MMs for each node in turn)
    if DISCOVER_MIXING_EACH_NODE:
        hypparams = dict(
            mixing_type=mixing_ty,
            truths=truths, oracle_Z=ORACLE_Z, oracle_K=ORACLE_K, oracle_G=ORACLE_G, k_max=options.KMAX, vb=0,
            lg=options.logger)

        # truths.dg.plot_X(data)
        ours = CausalChangeTopological(**hypparams)

        # MM models: one mixture per node was discovered (ignoring the graph)
        if mixing_ty.is_unconditional_mixture():
            ours.fit_latent_discrete_given_DAG(given_A.copy(), SKIP_PRUNING,
                                               skip_sets=True)  # skip set aggregation using MI stuff which is part of our approach- here only MMs
            e_Z_n = ours.e_Z_n
            # extract "intervention targets": each node that has more than one cluster
            # e_n_Z = [set(nodei) for nodei in range(e_Z_n) if len(np.unique(e_Z_n[nodei]))>1]
            e_n_Z = [set([node for node in range(len(e_Z_n)) if len(np.unique(e_Z_n[node])) > 1])]
            metrics_mixing = compare_Z(X.shape[0], truths['t_A'], None,
                                       truths['t_Z'], truths['t_n_Z'], None, None, e_n_Z, e_Z_n, None, exp["GS"])
        else:
            ours.fit_latent_discrete_given_DAG(given_A.copy(), SKIP_PRUNING)  # copy as pruning modifies it
            e_Z, Z_pairs, e_n_Z, e_Z_n, pprobas = ours.e_Z, ours.Z_pairs, ours.e_n_Z, ours.e_Z_n, ours.pprobas
            metrics_mixing = compare_Z(X.shape[0], truths['t_A'], nx.to_numpy_array(ours.graph_state),
                                       truths['t_Z'], truths['t_n_Z'], e_Z, Z_pairs, e_n_Z, e_Z_n, pprobas, exp["GS"])
    # MM over all variables
    else:
        range_k = range(1, options.KMAX + 1) if not ORACLE_K else params["K"]
        true_global_idl = None
        idl, pproba, _ = fit_mixture_model(mixing_ty, X, range_k, true_global_idl)
        e_n_Z = [set([node for node in range(X.shape[1])]) if len(np.unique(idl)) > 1 else set()]
        e_Z_n = [idl for _ in range(X.shape[1])]
        metrics_mixing = compare_Z(X.shape[0], truths['t_A'], None,
                                   truths['t_Z'], truths['t_n_Z'], None, None, e_n_Z, [idl for _ in range(len(X))],
                                   [pproba for _ in range(len(X))], exp["GS"])

    idf = oracle_ty.value + mixing_ty.value if not DISCOVER_G else mth.nm() + oracle_ty.value + mixing_ty.value
    if options.visu and '_dg' in truths: truths['_dg'].plot_X_idls(X, e_Z_n, plot_dir, idf)

    run_info(f"\tEstimated Targets: {e_n_Z}, True: {truths['t_n_Z']}", options.logger, options.verbosity)
    if mixing_ty.is_unconditional_mixture():
        run_info(
            f"\tEval: ami-n {metrics_mixing['node-amis']:.2f}\tF1-iv {metrics_mixing['f1-iv']:.2f} (fp {metrics_mixing['fp-iv']:.2f}\tfn {metrics_mixing['fn-iv']:.2f})\t{[f'{met}: {vl:.2f}' for met, vl in metrics_mixing.items()]}",
            options.logger, options.verbosity)
    else:
        run_info(
            f"\tEval: \tami-n {metrics_mixing['node-amis']:.2f}\tJacc {metrics_mixing['jacc']:.2f}\tF1-pair {metrics_mixing['f1-pair']:.2f}\tF1-iv {metrics_mixing['f1-iv']:.2f} (fp {metrics_mixing['fp-iv']:.2f}\tfn {metrics_mixing['fn-iv']:.2f})\t{[f'{met}: {vl:.2f}' for met, vl in metrics_mixing.items()]}",
            options.logger, options.verbosity)
    run_info(f"", options.logger, options.verbosity)
    results[idf] = dict(mth=idf, metrics=metrics_mixing)
    return results
