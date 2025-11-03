
def run_get_plot_filename(options, exp, case, rep):
    return str(options.plot_dir) + str(options.exp_type) + "_" + '_'.join \
        ([f"{e}_{exp[e]}" for e in exp]) + f"/{case}/{rep}/"

def run_info(str, lg, vb):
    if vb > 0:
        lg.info(str)
