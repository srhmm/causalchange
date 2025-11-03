from collections import defaultdict

import networkx as nx

from src.exp.exp_change.algos import CD, OracleType
from src.exp.exp_change.config_mix import Options
from src.exp.exp_change.gen.generate import gen_interventional_mixture
from src.exp.exp_change.run.run_mixseparation import run_dgen
from src.exp.exp_change.util.run_util import run_info, run_get_plot_filename
from src.causalchange.mixing.mixing import MixingType
from src.causalchange.util.util import compare_Z


def run_case_interventional_mixture(options, params, case, exp, rep_seed, sub_seed):
    plot_dir = run_get_plot_filename(options, exp, case, rep_seed[0] + 1)
    X, truths = gen_interventional_mixture(params)

    results = defaultdict(dict)
    type_of_oracle_knowledge = options.get_oracles()
    type_of_causal_discovery_algos = options.methods
    mixing_ty = MixingType.MIX_LIN
    for method_ty in type_of_causal_discovery_algos:
        if method_ty == CD.CausalMixtures:
            for oracle_ty in type_of_oracle_knowledge:
                results = run_cd_method(results, X, truths, method_ty, oracle_ty, mixing_ty, options, exp)
        else:
            results = run_cd_method(
                results, X, truths, method_ty, OracleType.SKIP, MixingType.SKIP, options, exp)
    return results


def run_case_causal_discovery(options: Options, params: dict, case, exp: dict, rep_seed, sub_seed):
    """ Run causal discovery and causal mixture discovery methods."""
    plot_dir = run_get_plot_filename(options, exp, case, rep_seed[0] + 1)
    X, truths = run_dgen(options, params, case, exp, rep_seed, sub_seed)

    results = defaultdict(dict)
    type_of_oracle_knowledge = options.get_oracles()
    type_of_causal_discovery_algos = options.methods

    type_of_mixing_algos = options.get_mixing_algos()

    for method_ty in type_of_causal_discovery_algos:
        if method_ty == CD.CausalMixtures:

            for mixing_ty in type_of_mixing_algos:
                for oracle_ty in type_of_oracle_knowledge:
                    results = run_cd_method(
                        results, X, truths, method_ty, oracle_ty, mixing_ty, options, exp)

        else:
            results = run_cd_method(
                results, X, truths, method_ty, OracleType.SKIP, MixingType.SKIP, options, exp)

    return results


def run_cd_method(results, X, truths, method_ty: CD, oracle_ty: OracleType, mixing_ty:MixingType, options: Options, exp: dict):
    run_info(f'\tMethod: {method_ty.value}', options.logger, options.verbosity)
    DISCOVERS_MIXING = method_ty.discovers_mixture_assignments()
    DISCOVERS_CHANGES = method_ty.discovers_context_changes()

    intv_args_dict, args = truths.get("intv_args_dict", None), truths.get("args", None)
    if method_ty == CD.MixtureUTIGSP: assert intv_args_dict is not None and args is not None

    # Causal discovery
    cls = method_ty.get_method()
    oracle_Z = oracle_ty.is_Z_known()
    kwargs = dict(
        truths=truths, args=args, intv_args_dict=intv_args_dict, lg=options.logger, oracle_Z=oracle_Z, mixing_type=mixing_ty,
        k_max=options.KMAX, vb=options.verbosity)
    kwargs["pruning_G"] = True
    cls.fit(X, **kwargs)

    # CD eval
    true_nxg = truths['true_g']
    metrics_cd = cls.get_graph_metrics(true_nxg)

    run_info("", options.logger, options.verbosity)
    run_info(f"\t***\tCD: {method_ty}", options.logger, options.verbosity)
    run_info(
        f"\tEval Graph: \tF1: {metrics_cd['f1']:.2f}, TP: {metrics_cd['tp']} FP: {metrics_cd['fp']}, SC: {metrics_cd['sc']:.2f}, SD: {metrics_cd['sd']:.2f}, SHD: {metrics_cd['shd-nm']:.2f}",
        options.logger, options.verbosity)

    # MEchanism change eval
    if DISCOVERS_CHANGES:
        metrics_avg = cls.model.get_metrics_changes(truths["mechanism_partitions"])
        run_info(f"\tEval Changes: f1-pair={metrics_avg['macro_pairwise_f1']}, ami={metrics_avg['macro_ami']}, var-f1={metrics_avg['macro_variant_pairwise_f1']}, invar-f1={metrics_avg['macro_invariant_pairwise_f1']}", options.logger, options.verbosity)

        metrics_cd.update(metrics_avg)
    #    assert cls.p_C_n is not None #partition of c per node
    #    raise NotImplementedError
    # Mixture eval
    if DISCOVERS_MIXING:
        assert cls.e_Z_n is not None and cls.e_n_Z is not None
        metrics_mixing = compare_Z(
            X.shape[0], truths['t_A'], nx.to_numpy_array(cls.dag),
            truths['t_Z'], truths['t_n_Z'], None, None, cls.e_n_Z, cls.e_Z_n, None, exp["GS"])
        run_info(
            f"\tEval Mixing: \tJacc {metrics_mixing.get('jacc', -1):.2f} AMI {metrics_mixing.get('node-amis', -1):.2f} F1-iv {metrics_mixing.get('f1-iv', -1):.2f} \t{[f'{met}: {vl:.2f}' for met, vl in metrics_mixing.items()]}",
            options.logger, options.verbosity)
        run_info(f"\tEstimated Mixing Targets: {cls.e_n_Z}, True: {truths['t_n_Z']}", options.logger, options.verbosity)

        metrics_cd.update(metrics_mixing)

    nm_cd = method_ty.value + str(oracle_ty)+ str(mixing_ty)
    run_info(oracle_ty, options.logger, options.verbosity)
    results[nm_cd] = dict(mth=nm_cd, metrics=metrics_cd)
    return results

