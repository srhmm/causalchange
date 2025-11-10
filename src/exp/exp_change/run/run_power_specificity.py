import networkx as nx
import numpy as np

from collections import defaultdict

from src.exp.exp_change.run.run_mixseparation import run_dgen
from src.exp.exp_change.util.run_util import run_info, run_get_plot_filename
from src.causalchange.causal_change_topo import CausalChangeTopological
from src.causalchange.util.util import compare_Z, lmg_to_directed_edge_adj, nxdigraph_to_lmg, compare_lmg_DAG


def run_case_power_specificity(options, params, case, exp, rep_seed, sub_seed):
    X, truths = run_dgen(options, params, case, exp, rep_seed, sub_seed)
    type_of_oracle_knowledge = options.get_oracles()
    type_of_mixing_algos = options.get_mixing_algos()  # only ours rn
    type_of_causal_discovery_algos = options.methods

    results = defaultdict(dict)
    for method_ty in type_of_causal_discovery_algos:
        for oracle_ty in type_of_oracle_knowledge:
            results = run_case_power_specificity_methods(results, X, truths, options, params, case, exp, method_ty,
                                                         oracle_ty, rep_seed)
    return results


def run_case_power_specificity_methods(results, X, truths, options, params, case, exp, method_ty, oracle_ty, rep_seed):
    baseline = method_ty
    seed = rep_seed[1].generate_state(1)[0]
    np.random.seed(seed)
    plot_dir = run_get_plot_filename(options, exp, case, rep_seed[0] + 1)
    t_A, t_n_Z, t_Z = truths['t_A'], truths['t_n_Z'], truths['t_Z']

    # comparison: regenerate data without any confounding, under the same DAG
    X_deconfd = truths["_dg"].gen_unconfounded_X()  # empties dg.conf_ind_sets in truths["_dg"]
    # dg.plot_X(X, "debug/confd_")
    # dg.plot_X(X_deconfd, "debug/deconfd_")

    # %% BASELINE 1 another dataset over the same DAG with confounders removed
    cls = baseline.get_method()
    nm = baseline.value
    cls.fit(X_deconfd, truths=dict())

    base_deconfd_lmg = cls.get_labelled_mixed_graph()
    base_deconfd_dag = lmg_to_directed_edge_adj(base_deconfd_lmg)
    true_nxg = truths['true_g']
    metrics_deconfd = cls.get_graph_metrics(true_nxg)

    # %% BASELINE 2 data with confounders/mixing
    cls = baseline.get_method()
    cls.fit(X, truths=dict())
    metrics_confd = cls.get_graph_metrics(true_nxg)

    base_confd_lmg = cls.get_labelled_mixed_graph()
    base_confd_dag = lmg_to_directed_edge_adj(base_confd_lmg)

    # %% OURS
    oracle_Z = oracle_ty.is_Z_known()
    hypparams = dict(truths=truths, oracle_Z=oracle_Z, oracle_K=False, oracle_G=False, k_max=options.KMAX,
                     vb=options.verbosity - 1, lg=options.logger)
    top = CausalChangeTopological(**hypparams)
    top.fit_Z_given_G(base_confd_dag)  # currently only consider directed edges

    graph = top.graph_state
    our_dag = nx.to_numpy_array(graph)
    est_lmg = nxdigraph_to_lmg(graph)
    true_lmg = nxdigraph_to_lmg(true_nxg)
    metrics_pruned = compare_lmg_DAG(true_lmg, est_lmg)

    # %% METRICS
    run_info("", options.logger, options.verbosity)
    run_info(f"\t***\tBaseline: {baseline}", options.logger, options.verbosity)
    run_info(
        f"\tNo mixing, \tF1: {metrics_deconfd['f1']:.2f}, TP: {metrics_deconfd['tp']} FP: {metrics_deconfd['fp']}",
        options.logger, options.verbosity)
    run_info(
        f"\tWith mixing, \tF1: {metrics_confd['f1']:.2f}, TP: {metrics_confd['tp']} FP: {metrics_confd['fp']}",
        options.logger, options.verbosity)
    run_info(
        f"\tAfter pruning, \tF1: {metrics_pruned['f1']:.2f}, TP: {metrics_pruned['tp']} FP: {metrics_pruned['fp']}",
        options.logger, options.verbosity)

    run_info(f"\tEstimated Targets: {top.e_n_Z}, True: {truths['t_n_Z']}", options.logger, options.verbosity)

    # %% DIRECT COMPARISON
    # comparison 1
    idf = 'deconf-conf'
    compA1, compA2 = base_deconfd_dag, base_confd_dag
    metrics_comp_A, metrics_deconfd, metrics_confd = compare_graph_metrics(t_A, compA1, compA2, metrics_deconfd,
                                                                           metrics_confd, idf)

    # comparison 2
    idf = 'conf-pruned'
    compA1, compA2 = base_confd_dag, our_dag
    metrics_comp_B, metrics_confd, metrics_pruned = compare_graph_metrics(t_A, compA1, compA2, metrics_confd,
                                                                          metrics_pruned, idf)

    # info: comparisons
    run_info(f"\tSupergraph discov: \tPR' {metrics_comp_A['pr-adjusted-g']:.2f}", options.logger, options.verbosity)
    run_info(
        f"\tIdentif improv: \tTPR {metrics_comp_A['tpr']:.2f}\t{[f'{met}: {vl:.2f}' for met, vl in metrics_comp_A.items()]}",
        options.logger, options.verbosity)
    run_info(
        f"\tPruning improv: \tFPR' {metrics_comp_B['fpr-adjusted-g']:.2f}\tSHD {metrics_comp_B['shd']:.2f} \t{[f'{met}: {vl:.2f}' for met, vl in metrics_comp_B.items()]}",
        options.logger, options.verbosity)
    metrics_mixing = compare_Z(X.shape[0], truths['t_A'], nx.to_numpy_array(top.graph_state),
                               truths['t_Z'], truths['t_n_Z'], top.e_Z, top.Z_pairs, top.e_n_Z, top.e_Z_n, top.pprobas)
    run_info(
        f"\tMixing Eval: \tJacc {metrics_mixing['jacc']:.2f} \t{[f'{met}: {vl:.2f}' for met, vl in metrics_mixing.items()]}",
        options.logger, options.verbosity)
    metrics_pruned.update(metrics_mixing)

    idf = method_ty.value + '-' + oracle_ty.value
    results[idf + '-base'] = dict(mth=idf + '-base', metrics=metrics_confd)
    results[idf + '-nocfd'] = dict(mth=idf + '-nocfd', metrics=metrics_deconfd)
    results[idf + '-ours'] = dict(mth=idf + '-ours', metrics=metrics_pruned)
    results[idf + '-improv-identif'] = dict(mth=idf + '-improv-identif', metrics=metrics_comp_A)
    results[idf + '-improv-pruning'] = dict(mth=idf + '-improv-pruning', metrics=metrics_comp_B)

    return results


def compare_graph_metrics(t_A, base_A, comp_A, base_metrics, comp_metrics, idf):
    HIGHER_BETTER = ['f1', 'mcc', 'tp', 'pr', 'tpr', 'tnr']
    LOWERBETTER = ['sd', 'sc', 'shd', 'fp', 'fpr', 'fnr']

    improv = {}
    for ky in ['f1', 'tp', 'fp', 'pr', 'tpr', 'tnr', 'fpr', 'fnr', 'sd', 'sc', 'shd']:
        assert ky in HIGHER_BETTER or ky in LOWERBETTER and not ky in HIGHER_BETTER and ky in LOWERBETTER
        improv[ky] = comp_metrics[ky] - base_metrics[ky] if ky in HIGHER_BETTER else base_metrics[ky] - comp_metrics[ky]

    # Adjustment of true negatives due to imbalanced TN/FP
    # fp can only turn into tn or remain fp -> disregard edges marked in both cases as tn
    overlap_tn = sum([sum([1 if (base_A[i][j] == 0 and comp_A[i][j] == 0 and t_A[i][j] == 0) else 0
                           for j in range(len(t_A[i]))]) for i in range(len(t_A))])

    comp_nn = comp_metrics['tn'] - overlap_tn + comp_metrics['fp']
    base_nn = base_metrics['tn'] - overlap_tn + base_metrics['fp']

    def safediv(a, b):
        if b == 0: return 0
        return a / b

    comp_pr = safediv((comp_metrics['tp'] + comp_metrics['fp']),
                      (comp_metrics['tp'] + comp_metrics['fp'] + comp_metrics['tn'] + comp_metrics['fn'] - overlap_tn))
    base_pr = safediv((base_metrics['tp'] + base_metrics['fp']),
                      (base_metrics['tp'] + base_metrics['fp'] + base_metrics['tn'] + base_metrics['fn'] - overlap_tn))

    comp_fpr_type1 = 0 if comp_nn == 0 else safediv(comp_metrics['fp'], comp_nn)
    base_fpr_type1 = 0 if base_nn == 0 else safediv(base_metrics['fp'], base_nn)

    improv['pr-adjusted-g'] = comp_pr - base_pr
    improv['fpr-adjusted-g'] = base_fpr_type1 - comp_fpr_type1
    improv['tnr-adjusted-g'] = improv['fpr-adjusted-g']

    # adjustment depends on which methods are being compared to one another, therefore insert new entries into metrics (with unique identifier)
    cm = comp_metrics.copy()
    cm[f'{idf}-pr-adjusted-g'] = comp_pr
    cm[f'{idf}-fpr-adjusted-g'] = comp_fpr_type1
    cm[f'{idf}-tnr-adjusted-g'] = 1 - comp_fpr_type1

    bm = base_metrics.copy()
    bm[f'{idf}-pr-adjusted-g'] = base_pr
    bm[f'{idf}-fpr-adjusted-g'] = base_fpr_type1
    bm[f'{idf}-tnr-adjusted-g'] = 1 - base_fpr_type1
    # nm identifies which base metrics we compare against, include in comp metrics if we compare to different baselines (algo, idf...)

    return improv, bm, cm
