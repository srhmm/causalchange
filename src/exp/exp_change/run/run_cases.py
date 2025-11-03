import os

from joblib import Parallel, delayed
from numpy.random import SeedSequence

from src.exp.exp_change.algos import OracleType
from src.exp.exp_change.config_mix import ExpType
from src.exp.exp_change.gen.generate import IvMode
from src.exp.exp_change.run.run_cd import run_case_causal_discovery, run_case_interventional_mixture
from src.exp.exp_change.run.run_mixseparation import run_case_mixseparation
from src.exp.exp_change.run.run_power_specificity import run_case_power_specificity
from src.exp.exp_change.util.case_results import CaseReslts, write_cases
from src.exp.exp_change.util.run_util import run_info
from src.causalchange.mixing.mixing import MixingType


def run_case_safe(options, params, case, exp, rep_seed, seed_seq):
    try:
        return _run_case(options=options, params=params, case=case, exp=exp, rep_seed=rep_seed, sub_seed=1)
    except Exception as e:
        options.logger.info(f"Error occured {e}, retry")
        cs = seed_seq.spawn(100)
        for sub_seed in enumerate(cs):
            options.logger.info(f"Repeat with seed={sub_seed}")
            try:
                return _run_case(options=options, params=params, case=case, exp=exp, rep_seed=rep_seed,
                                 sub_seed=sub_seed)
            except Exception:
                continue
    try:
        return _run_case(options=options, params=params, case=case, exp=exp, rep_seed=rep_seed, sub_seed=1)
    except Exception as e:
        options.logger.info(f"Error occured {e}")
        raise e


def _run_case(**kwargs):
    """ Run one repetition of a experiment depending on exp type """

    options = kwargs["options"]
    if options.exp_type in [ExpType.CLUSTERING, ExpType.CLUSTERING_GRAPHSTRUCTURES]:
        result = run_case_mixseparation(**kwargs)
    elif options.exp_type == ExpType.POWER_SPECIFICITY:
        result = run_case_power_specificity(**kwargs)
    elif options.exp_type == ExpType.CAUSAL_DISCOVERY:
        result = run_case_causal_discovery(**kwargs)
    elif options.exp_type == ExpType.INTERVENTIONAL_MIXTURE:
        result = run_case_interventional_mixture(**kwargs)
    else:
        raise ValueError
    # run_info(f'\tFinished Case: {case} for methods {result.keys()}', options.logger, options.verbosity)
    return result


def run_cases(options):
    exps = options.get_experiments()
    cases = options.get_cases()
    reslts = [CaseReslts(case) for case in cases]

    # writing results only
    if options.read:
        for exp in exps:
            read_dir = options.out_dir  + '_'.join([f"{ky}_{str(vl)}" for (ky, vl) in exp.items()]) +'/'
            for attr in options.fixed:
                write_cases(options, exp, attr, read_dir)
        return

    # Each experiment configuration (eg linear Gaussian)
    out_dir = options.out_dir
    for exp in exps:
        options.out_dir = out_dir  + '_'.join([f"{ky}_{str(vl)}" for (ky, vl) in exp.items()]) +'/'
        run_info("", options.logger, options.verbosity)
        run_info("*** Experiment ***", options.logger, options.verbosity)
        run_info(f"Experiment Type: {options.exp_type.long_nm()}", options.logger, options.verbosity)
        run_info(f"Fixed Parameters: { '_'.join([f'{ky}_{str(vl)}' for (ky, vl) in exp.items()])}", options.logger, options.verbosity)
        run_info(f"Base case: {options.get_base_attribute_idf()}", options.logger, options.verbosity)
        nln = '\n\t'
        run_info(f'All cases:\n\t{nln.join(cases)}', options.logger, options.verbosity)
        run_info(f"All CD methods: {', '.join([str(m) for m in options.methods])}", options.logger, options.verbosity)
        run_info(f"All clustering methods: {', '.join(['our MLR' if m==MixingType.MIX_LIN else str(m) for m in options.get_mixing_algos()])}", options.logger, options.verbosity)
        run_info(f"Oracles, if any: {', '.join([ 'without oracles' if m==OracleType.hatGhatZ else str(m) for m in options.get_oracles()])}", options.logger, options.verbosity)
        run_info("", options.logger, options.verbosity)

        # Each parameter configuration (eg N=10 nodes, Z=2 cfds, ...)
        for case, res in zip(cases, reslts):
            if check_case_exists(options, case, exp, [m.value for m in options.methods]):

                continue
            run_info(f"CASE: {case}", options.logger, options.verbosity)

            ss = SeedSequence(options.seed)
            cs = ss.spawn(options.reps)
            params = cases[case]
            if exp["IVM"] == IvMode.MULTI_CONTEXT.value: params["S"] = params["C"] * params["S"]  # s means per-context samples here!
            run_one_rep = lambda rep: (
                run_case_safe(options, params, case, exp, rep, ss) if options.safe
                else _run_case(options=options, params=params, case=case, exp=exp, rep_seed=rep, sub_seed=1))

            if options.n_jobs > 1:
                results = Parallel(n_jobs=options.n_jobs)(delayed(
                    run_one_rep)(rep_seed) for rep_seed in enumerate(cs))
            else:
                results = [run_one_rep(rep_seed) for rep_seed in enumerate(cs)]

            res.add_reps(results)
            res.write_case(params, exp, options)

        for attr in options.fixed:
            write_cases(options, exp, attr)

def check_case_exists(options, case, exp, methods):
    exists_all=True
    options.logger.info("")
    options.logger.info(f"Case: { case} exists")

    base_attribute_idf = options.get_base_attribute_idf()
    long_path_pre = os.path.join(options.out_dir, os.path.join(
        str(options.exp_type) + "_" + '_'.join([f"{e}_{exp[e]}" for e in exp])), base_attribute_idf)
    path_pre = os.path.join(options.out_dir, base_attribute_idf)

    path = os.path.join(path_pre, os.path.join("tikzfiles", "all/"))
    os.makedirs(path, exist_ok=True)

    for mth in methods:
        fl = os.path.join(path, f"{case}_m_{mth}.tsv")
        exists_all = exists_all and os.path.exists(fl)
        #if os.path.exists(fl):
        #    print("Exists:", fl)
        #else: print("Not exists:", fl)
    run_info("Case exists:"+str(exists_all )+"at"+str(fl), options.logger, options.verbosity)
    return exists_all