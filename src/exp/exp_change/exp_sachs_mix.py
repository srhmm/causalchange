import matplotlib
import networkx as nx
import numpy as np

from src.exp.exp_change.algos import CD
from src.exp.exp_change.config_mix import ExpType
from src.exp.exp_change.gen.sachs.sachs_utils import generate_mixture_sachs
from src.exp.exp_change.run.run_sachs_mix import sachs_run_causal_discovery

if __name__ == "__main__":
    """
    >>> python exp_sachs.py -m 0
    """
    import argparse

    matplotlib.use('Agg')

    ap = argparse.ArgumentParser("RUN")

    def enum_help(enm) -> str: return ','.join([str(e.value) + '=' + str(e) for e in enm])

    # experiment
    ap.add_argument("-m", "--mode", default=0, help=f"{enum_help(ExpType)}", type=int)
    t_Z, t_n_Z, intv_args_dict, mixture_samples, num_nodes, A, idx2var_dict = generate_mixture_sachs()
    context_samples = {ci: np.array(intv_args_dict[c]["samples"]) for ci, c in enumerate( intv_args_dict)}

    true_g = nx.from_numpy_array(A.T, create_using=nx.DiGraph)
    intv_targets = [("Akt", 3), ("PKC", 4), ("PIP2", 5), ("Mek", 6), ("PIP3", 7)]

    #%% Table 1, 2
    # sachs_run_mixtureutigsp_ours(mixture_samples, intv_args_dict, intv_targets, t_Z, idx2var_dict)

    #%%  Causal Discovery
    #KMAX = 5
    #CD_ALGOS = [CD.CausalMixtures]
    #for cd_method in CD_ALGOS:
    #    sachs_run_causal_discovery(mixture_samples, cd_method, KMAX)
    CD_ALGOS = [CD.TopicContexts]
    for cd_method in CD_ALGOS:
        sachs_run_causal_discovery(context_samples, cd_method, None, idx2var_dict, true_g)
