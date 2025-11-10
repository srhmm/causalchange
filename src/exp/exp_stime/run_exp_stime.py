from src.exp.exp_stime.options import Options
from src.exp.exp_stime.utils.gen_timeseries import Func, Noise
from src.stime import DiscrepancyTestType
from src.exp.exp_stime.run import run
from src.stime import MethodType

if __name__ == "__main__":
    import sys
    import argparse
    import logging
    from pathlib import Path

    ap = argparse.ArgumentParser("SPCTME")

    def enum_help(enm) -> str:
        return ','.join([str(e.value) + '=' + str(e) for e in enm])

    # Experiment Parameters
    ap.add_argument("-exp", "--exp", default='ALL', help="parameter to vary: T,N,C,R,D,CPS,I,IVS,ALL")

    ap.add_argument("-rep", "--reps", default=20, help="repetitions of experiment", type=int)
    ap.add_argument("-met", "--metrics", default=['f1', 'mcc', 'shd', #'sid',
                                                  'tpr', 'fpr', 'fdr', 'tp', 'fp', 'tn', 'fn'],
                    nargs="+", help="f1, sid, shd, mcc, tpr, fpr",
                    type=int)
    ap.add_argument("-sd", "--seed", default=42, help="seed", type=int)
    ap.add_argument("-nj", "--n_jobs", default=1, help="n parallel jobs", type=int)

    # Iterable Data Generation Parameter
    # observed
    ap.add_argument("-t", "--timelen", default=[200], nargs="+",
                    help="time series length (each context, all regimes)", type=int)
    ap.add_argument("-d", "--datasets", default=[1], nargs="+", help="number of datasets (each with time series len)",
                    type=int)
    ap.add_argument("-n", "--nodes", default=[5], nargs="+", help="num nodes", type=int)
    # unobserved
    ap.add_argument("-cps", "--cutpoints", default=[2], nargs="+", help="num regimes", type=int)
    ap.add_argument("-r", "--regimes",default=[2],  #default=[2, 3, 4, 5, 6, 8],
                    nargs="+", help="num regimes", type=int)

    ap.add_argument("-c", "--contexts", #default=[2, 5, 10, 15],
                    default=[2], nargs="+", help="num contexts", type=int)

    ap.add_argument("-f", "--functions_F", default=[1], nargs="+", help=f"{enum_help(Func)}", type=int)
    ap.add_argument("-ns", "--noises_NS", default=[0], nargs="+", help=f"{enum_help(Noise)}", type=int)

    ap.add_argument("-i", "--interventions",default=[  0.5 ], # default=[0, 0.2, 0.5, 0.8, 1],
                    nargs="+",
                    help="num intervened nodes per context/regime", type=int)
    ap.add_argument("-str", "--intervention_strength", default=[50], nargs="+",
                    help="strength of coefficient shifts, .1 if arg=10 and so on", type=int)


    # Base Parameters while iterating over the others
    ap.add_argument("-tbs", "--timelen_base", default=1000, help="time series length (each context, all regimes)",
                    type=int)
    ap.add_argument("-dbs", "--datasets_base", default=1, help="number of datasets (each with time series len)",
                    type=int)
    ap.add_argument("-nbs", "--nodes_base", default=5, help="num nodes", type=int)
    ap.add_argument("-rbs", "--regimes_base", default=3, help="num regimes", type=int)
    ap.add_argument("-cpsbs", "--cutpoints_base", default=2, help="num cutpoints=num chunks-1", type=int)
    ap.add_argument("-cbs", "--contexts_base", default=2, help="num contexts", type=int)
    ap.add_argument("-ibs", "--interventions_base", default=0.5, help="fraction of intervened arcs", type=int)
    ap.add_argument("-strbs", "--intervention_strength_base", default=50, nargs="+",
                    help="strength of coefficient shifts, .1 if arg=10 and so on", type=int)

    # Constant Data Generation Parameters
    min_dur = 20
    # ap.add_argument("-in", "--intervened_nodes", default=2, help="num intervened nodes per context", type=int)
    ap.add_argument("-ttm", "--true_tau_max", default=2, help="tau max in data generatio"
                                                              "n", type=int)
    ap.add_argument("-tmd", "--true_min_dur", default=min_dur, type=int)
    ap.add_argument("-rd", "--regime_drift", default=False, type=bool)
    ap.add_argument("-hi", "--hard_intervention", default=False, type=bool)

    ap.add_argument("--quick", action="store_true", help="run a shorter experiment for testing")
    ap.add_argument("--safe", action="store_true", help="catch exceptions and skip experiment cases")

    # Method Parameters
    #ap.add_argument("-m", "--methods", default=implemented_methods, nargs="+", help=f"{enum_help(MethodType)}",
    #                type=int)
    ap.add_argument("-dt", "--discrep_test", default=1, help=f"{enum_help(DiscrepancyTestType)}", type=int)
    ap.add_argument("-atm", "--assumed_tau_max", default=2, help="tau max that our algo assumes", type=int)
    ap.add_argument("-amd", "--assumed_min_dur", default=min_dur, type=int)
    ap.add_argument("-v", "--verbosity", default=1, help='use >1 to see output of dag search', type=int)

    # Path
    ap.add_argument("-bd", "--base_dir", default="")
    ap.add_argument("-wd", "--write_dir", default="res/res_stime/")

    argv = sys.argv[1:]
    nmsp = ap.parse_args(argv)


    logging.basicConfig()
    log = logging.getLogger("SPCTME-"+nmsp.exp)
    log.setLevel("INFO")

    nmsp.methods = [#MethodType.PCMCIPLUS.value,
                      MethodType.ST_GP.value,
                       ##MethodType.GP_HYBRID.value,  #MethodType.GP_QFF.value,
                 #MethodType.GP_REGIMES.value, MethodType.GP_DAG.value,
                # MethodType.RPCMCI.value,  #methodType.RPCMCI_REGIMES.value,#
               # MethodType.JPCMCI.value, #MethodType.JPCMCI_REGIMES.value,#
                #        MethodType.VARLINGAM.value, #MethodType.VARLINGAM_REGIMES.value,#
                 #  MethodType.CDNOD.value,  #MethodType.CDNOD_REGIMES.value,#
                  #  MethodType.DYNOTEARS.value,  #MethodType.DYNOTEARS_REGIMES.value,#
                           ]

    # store parameters
    options = Options(#discrepancy_test=DiscrepancyTestType(nmsp.discrep_test),
                         methods=[m for m in map(MethodType, nmsp.methods)],
                         metrics=nmsp.metrics,
                         functions=[Func.id_to_func(m) for m in  nmsp.functions_F],
                         noises=[Noise.id_to_noise(m) for m in  nmsp.noises_NS],
                         contexts_C=nmsp.contexts if nmsp.exp in ['ALL', 'C'] else [nmsp.contexts_base],
                         regimes_R=nmsp.regimes if nmsp.exp in ['ALL', 'R'] else [nmsp.regimes_base],
                         cutpoints_CPS=nmsp.cutpoints if nmsp.exp in ['ALL', 'CPS'] else [nmsp.cutpoints_base],
                         nodes_N=nmsp.nodes if nmsp.exp in ['ALL', 'N'] else [nmsp.nodes_base],
                         datasets_D=nmsp.datasets if nmsp.exp in ['ALL', 'D'] else [nmsp.datasets_base],
                         timelen_T=nmsp.timelen if nmsp.exp in ['ALL', 'T'] else [nmsp.timelen_base],
                         interventions_I=nmsp.interventions if nmsp.exp in ['ALL', 'I'] else [nmsp.interventions_base],
                         intervention_strength_S=nmsp.intervention_strength if nmsp.exp in ['ALL', 'IVS'] else [nmsp.intervention_strength_base],
                         contexts_C_base=nmsp.contexts_base,
                         regimes_R_base=nmsp.regimes_base,
                         cutpoints_CPS_base=nmsp.cutpoints_base,
                         nodes_N_base=nmsp.nodes_base,
                         datasets_D_base=nmsp.datasets_base,
                         timelen_T_base=nmsp.timelen_base,
                         interventions_I_base=nmsp.interventions_base,
                         intervention_strength_S_base=nmsp.intervention_strength_base,
                         true_tau_max=nmsp.true_tau_max,
                         assumed_tau_max=nmsp.assumed_tau_max,
                         regime_drift=nmsp.regime_drift,
                         logger=log,
                         reps=nmsp.reps,
                         quick=nmsp.quick,
                         safe=nmsp.safe,
                         hard_intervention=nmsp.hard_intervention,
                         true_min_dur=nmsp.true_min_dur,
                         assumed_min_dur=nmsp.assumed_min_dur,
                         enable_SID_call='sid' in nmsp.metrics,
                         n_jobs=nmsp.n_jobs,
                         seed=nmsp.seed,
                         verbosity=nmsp.verbosity)

    # Logging
    out_dir = nmsp.write_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"{out_dir}run_exp_stime.log")
    fh.setLevel(logging.INFO)
    options.logger.addHandler(fh)
    options.out_dir = out_dir

    #options.logger_dag = logging.getLogger("DAGSEARCH")
    #options.logger_dag.setLevel("INFO")
    #fh = logging.FileHandler(f"{out_dir}run_dag.log")
    #fh.setLevel(logging.INFO)
    #options.logger_dag.addHandler(fh)

    import warnings
    warnings.filterwarnings("ignore")

    run(options)
