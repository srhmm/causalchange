import matplotlib

from src.exp.exp_change.config_contexts import OptionsContext, ExpType

from src.exp.exp_change.run.run_cases import run_cases

if __name__ == "__main__":
    """
    >>> python run_exp_topic.py -e 0
    """
    import sys
    import argparse
    import logging
    from pathlib import Path

    matplotlib.use('Agg')

    ap = argparse.ArgumentParser("RUN")

    def enum_help(enm) -> str:
        return ','.join([str(e.value) + '=' + str(e) for e in enm])

    vb_help = f"1: Experiment info (params, cases), \n2: Algo info (graph search) \n3: Score info (each edge)" #vb level may not always correct...

    # experiment
    ap.add_argument("-e", "--exp", default=2, help=f"{enum_help(ExpType)}", type=int)
    ap.add_argument("-sd", "--seed", default=42, type=int)
    ap.add_argument("-r", "--reps", default=10, type=int)

    # method parameters
    ap.add_argument("-k", "--kmax", default=5, help=f"max n clusters", type=int)

    # flags
    ap.add_argument("--safe", action="store_true", help="skip exceptions (regenerate datasets w new seed)")
    ap.add_argument("--demo", action="store_true", help="run fewer repetitions")
    ap.add_argument("--only_base_params", action="store_true", help="only base parameters, otherwise vary the parameters")
    ap.add_argument("--read", action="store_true", help="read results from --read_dir/$exp$/$params$/tikzfiles/all/ and aggregate to --read_dir/$exp$/$params$/tikzfiles/change/")
    ap.add_argument("--visu", action="store_true", help="plot to --plot_dir/")
    ap.add_argument("-nj", "--n_jobs", default=1, type=int)
    ap.add_argument("-v", "--verbosity", default=1, type=int, help=vb_help)

    # path
    ap.add_argument("-bd", "--base_dir", default="")
    ap.add_argument("-wd", "--write_dir", default="res_contexts/")
    ap.add_argument("-pd", "--plot_dir", default="plts_contexts/")
    ap.add_argument("-rd", "--read_dir", default="res_contexts/")

    argv = sys.argv[1:]
    nmsp = ap.parse_args(argv)

    logging.basicConfig()
    log = logging.getLogger(str(ExpType(nmsp.exp)))
    log.setLevel("INFO")

    options = OptionsContext(
        exp_type=ExpType(nmsp.exp),
        KMAX=nmsp.kmax,
        reps=nmsp.reps,
        n_jobs=nmsp.n_jobs,
        seed=nmsp.seed,
        logger=log,
        verbosity=nmsp.verbosity,
        read_dir=nmsp.read_dir,
        plot_dir=nmsp.plot_dir,
        onlybase=nmsp.only_base_params)
    #flags
    options.safe = nmsp.safe
    options.read = nmsp.read
    options.visu = nmsp.visu
    if nmsp.demo: options.reps = 3

    # Logging
    options.out_dir = nmsp.write_dir + f"res_e{nmsp.exp}/"
    options.read_dir = nmsp.read_dir + f"res_e{nmsp.exp}/"
    Path(options.out_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"{options.out_dir}run_e{nmsp.exp}.log")
    fh.setLevel(logging.INFO)
    options.logger.addHandler(fh)
    import warnings

    warnings.filterwarnings("ignore")

    run_cases(options)


