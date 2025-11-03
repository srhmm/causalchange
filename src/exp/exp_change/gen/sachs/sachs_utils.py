import os

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.metrics import adjusted_mutual_info_score


# sachs preprossing as in https://github.com/BigBang0072/mixture_mec/
def generate_mixture_sachs(fpath="../dsets/dsets_mixtures/sachs_yuhaow.csv"):
    df = pd.read_csv(fpath, delimiter="\t")

    intv_args_dict = {}
    mixture_samples = []
    mixture_idls = defaultdict(list)
    intv_targets = [("Akt", 3), ("PKC", 4), ("PIP2", 5), ("Mek", 6), ("PIP3", 7)]
    #Getting the names of the variables
    var_names = df.drop(columns=["experiment"]).columns.tolist()
    var2idx_dict = {var: idx for idx, var in enumerate(var_names)}
    idx2var_dict = {val: key for key, val in var2idx_dict.items()}
    #Getting the adjacecny matrix for this dataset
    A = get_sachs_adj_matrix(var2idx_dict)
    num_nodes = A.shape[0]

    #Adding the observational data
    print("Getting the observational data")
    intv_args_dict["obs"] = {}
    intv_args_dict["obs"]["tgt_idx"] = None
    obs_samples = df[(df["experiment"] == 1) | (df["experiment"] == 2)
                     ].drop(columns=["experiment"]).to_numpy()
    obs_indic = ((df["experiment"] == 1) | (df["experiment"] == 2)).astype(int).to_numpy()

    print("num samples: obs: ", obs_samples.shape[0])
    mixture_samples.append(obs_samples)

    for tgt_i in range(len(intv_targets)): mixture_idls[tgt_i].append(np.zeros(obs_samples.shape[0]))
    intv_args_dict["obs"]["samples"] = obs_samples
    intv_args_dict["obs"]["idl"] = obs_indic
    intv_args_dict["obs"]["true_params"] = dict(
        Si=np.cov(obs_samples, rowvar=False),
        mui=np.mean(obs_samples, axis=0),
        Ai=A,
    )
    t_n_Z = []
    for tgt_i, (tgt, expt_num) in enumerate(intv_targets):
        t_n_Z.append([var2idx_dict[tgt]])
        print("Getting the internvetional data: ", tgt)
        intv_samples = df[df["experiment"] == expt_num].drop(
            columns=["experiment"]).to_numpy()

        intv_indic = (df["experiment"] == expt_num).astype(int).to_numpy()
        print("num_samples: {}: {}".format(tgt, intv_samples.shape[0]))
        mixture_samples.append(intv_samples)


        for tgt_j in range(len(intv_targets)): mixture_idls[tgt_j].append(np.ones(intv_samples.shape[0]) if tgt_j == tgt_i else np.zeros(intv_samples.shape[0]))

        intv_args_dict[tgt] = {}
        intv_args_dict[tgt]["tgt_idx"] = var2idx_dict[tgt]
        #This will have clean mixture samples
        intv_args_dict[tgt]["samples"] = intv_samples
        intv_args_dict[tgt]["idl"] = intv_indic

        #Getting the new adjancecy matrix for this intervened dist (do intv)
        Ai = A.copy()
        Ai[var2idx_dict[tgt], :] = 0.0
        intv_args_dict[tgt]["true_params"] = dict(
            Si=np.cov(intv_samples, rowvar=False),
            mui=np.mean(intv_samples, axis=0),
            Ai=Ai,
        )

    #Acculmulating the samples in to one big matrix
    mixture_samples = np.concatenate(mixture_samples, axis=0)

    for tgt_i in range(len(intv_targets)): mixture_idls[tgt_i] = np.concatenate(mixture_idls[tgt_i] , axis=0)
    print("Total number of samples: ", mixture_samples.shape[0])
    t_Z = [arg for arg in mixture_idls.values()]

    return t_Z, t_n_Z, intv_args_dict, mixture_samples, num_nodes, A, idx2var_dict


def get_sachs_adj_matrix(var2idx_dict):
    '''
    '''
    num_nodes = len(var2idx_dict)
    A = np.zeros((num_nodes, num_nodes))
    #Now adding the edges
    A[var2idx_dict["Akt"], var2idx_dict["PKA"]] = 1.0
    A[var2idx_dict["Erk"], var2idx_dict["PKA"]] = 1.0
    A[var2idx_dict["Mek"], var2idx_dict["PKA"]] = 1.0
    A[var2idx_dict["Raf"], var2idx_dict["PKA"]] = 1.0
    A[var2idx_dict["JNK"], var2idx_dict["PKA"]] = 1.0
    A[var2idx_dict["p38"], var2idx_dict["PKA"]] = 1.0
    A[var2idx_dict["Akt"], var2idx_dict["PIP3"]] = 1.0
    A[var2idx_dict["PIP2"], var2idx_dict["PIP3"]] = 1.0
    A[var2idx_dict["PLCg"], var2idx_dict["PIP3"]] = 1.0
    A[var2idx_dict["PKC"], var2idx_dict["PIP2"]] = 1.0
    A[var2idx_dict["PIP2"], var2idx_dict["PLCg"]] = 1.0
    A[var2idx_dict["PKC"], var2idx_dict["PLCg"]] = 1.0
    A[var2idx_dict["Erk"], var2idx_dict["Mek"]] = 1.0
    A[var2idx_dict["Mek"], var2idx_dict["Raf"]] = 1.0
    A[var2idx_dict["Mek"], var2idx_dict["PKC"]] = 1.0
    A[var2idx_dict["Raf"], var2idx_dict["PKC"]] = 1.0
    A[var2idx_dict["JNK"], var2idx_dict["PKC"]] = 1.0
    A[var2idx_dict["p38"], var2idx_dict["PKC"]] = 1.0

    return A

def sachs_eval_iv_targets(e_Z, t_Z, intv_targets, idx2var_dict, intv_args_dict):
    mi_scores = {}
    for tgti, (nm, ivix) in enumerate(intv_targets):
        varix = [ky for ky in idx2var_dict if idx2var_dict[ky]==nm][0]
        assert idx2var_dict[varix]==nm
        assert nm in intv_args_dict
        score = adjusted_mutual_info_score(t_Z[tgti], e_Z[varix])
        print(f"{nm}: {score}")
        mi_scores[nm] = score
    return mi_scores

def sachs_plot_bivariate(node_x, node_y,  est_idl_j=None):
    import matplotlib.pyplot as plt
    t_Z, t_n_Z, intv_args_dict, mixture_samples, num_nodes, A, idx2var_dict = generate_mixture_sachs()
    true_g = nx.from_numpy_array(A.T, create_using=nx.DiGraph)
    plt.figure(figsize=(6, 5))

    for cls in np.unique(est_idl_j):
        plt.scatter(mixture_samples[:5846,node_x][est_idl_j==cls], mixture_samples[:5846,node_y][est_idl_j==cls], label=f'Class {cls}')
    plt.show()


def write_out_metrs(out_fl, metrics):
    os.makedirs(os.path.dirname(out_fl), exist_ok=True)
    write_file = open(out_fl, 'w')
    for met in metrics.keys(): write_file.write(f'{met}\t')
    write_file.write(f'\n')
    for met in metrics.keys(): write_file.write(f'{metrics[met]:.5f}\t')
    write_file.close()