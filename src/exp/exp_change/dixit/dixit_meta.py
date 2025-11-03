'''
Data loader codes are partly taken from https://github.com/csquires/utigsp
Processed data is also taken from https://github.com/csquires/utigsp/tree/master/real_data_analysis/dixit
for being consistent in comparisons

For Ground truth, Figure 4D of main paper Dixit et al. (https://pubmed.ncbi.nlm.nih.gov/27984732/) is taken
    true_edges_dixit_paper

Alternatively, Figure E2.a of IGSP paper (https://arxiv.org/abs/1705.10220) can be taken.
    true_edges_igsp_paper    

'''

import os
#from config import PROJECT_FOLDER, REALDATA_FOLDER
import numpy as np
import pandas as pd
import random
import causaldag as cd

DIXIT_FOLDER = os.path.join('', 'dixit')
DIXIT_DATA_FOLDER = os.path.join('', 'data')
DIXIT_ESTIMATED_FOLDER = os.path.join('', 'estimated')
DIXIT_FIGURES_FOLDER = os.path.join('', 'figures')


nnodes = 24

# note that these are for python indices, so they can be from 0 to 23
EFFECTIVE_NODES = [2, 9, 15, 16, 17, 20, 21, 22]

def dixit_get_samples():
    perturbations = np.load(os.path.join(DIXIT_FOLDER, 'data', 'perturbations.npy'))
    perturbation2ix = {p: ix for ix, p in enumerate(perturbations)}
    genes = np.load(os.path.join(DIXIT_FOLDER, 'data', 'genes.npy'))
    gene2ix = {g: ix for ix, g in enumerate(genes)}
    perturbation_per_cell = np.load(os.path.join(DIXIT_FOLDER, 'data', 'perturbation_per_cell.npy'))
    total_count_matrix = np.load(os.path.join(DIXIT_FOLDER, 'data', 'total_count_matrix.npy'))
    total_count_matrix = np.log1p(total_count_matrix)

    # === GET OBSERVATIONAL DATA
    control = 'm_MouseNTC_100_A_67005'
    control_cell_ixs = np.where(perturbation_per_cell == perturbation2ix[control])
    obs_samples = total_count_matrix[:, control_cell_ixs].squeeze().T

    setting_list = []
    for pnum, perturbation in enumerate(perturbations):
        if perturbation != control:
            iv_cell_ixs = np.where(perturbation_per_cell == perturbation2ix[perturbation])
            iv_samples = total_count_matrix[:, iv_cell_ixs].squeeze().T
            target_gene = perturbation[2:-2]
            setting_list.append({'known_interventions': {gene2ix[target_gene]}, 'samples': iv_samples})

    return obs_samples, setting_list


true_edges_dixit_paper = [(16, 1),
    (21, 1),
    (22, 1),
    (9, 2),
    (13, 2),
    (15, 2),
    (17, 2),
    (19, 2),
    (20, 2),
    (21, 2),
    (23, 2),    
    (2, 9),
    (13, 9),
    (15, 9),
    (17, 9),
    (19, 9),
    (20, 9),
    (22, 9),
    (23, 9),    
    (19, 10),
    (21, 10),
    (22, 10),
    (22, 11),
    (15, 12),
    (17, 12),
    (19, 12),
    (16, 13),
    (17, 16),
    (23, 16),
    (15, 20),
    (16, 20),
    (21, 20),
    (22, 20),
    (22, 21)]

true_B_dixit_paper = np.zeros((nnodes,nnodes))
for edge in true_edges_dixit_paper:
    true_B_dixit_paper[edge] = 1

true_edges_igsp_paper = [(2, 9),
    (2, 19),
    (2, 20),
    (9, 3),
    (9, 7),
    (9, 10),
    (9, 15),
    (15, 8),
    (15, 9),
    (15, 16),
    (16, 1),    
    (16, 2),
    (16, 9),
    (16, 13),
    (17, 18),
    (17, 20),
    (20, 7),
    (20, 16),
    (21, 5),    
    (21, 10),
    (21, 16),
    (21, 18),
    (22, 0),
    (22, 1),
    (22, 2),
    (22, 5),
    (22, 7),
    (22, 8),
    (22, 11),
    (22, 20),
    (22, 21)]

true_B_igsp_paper = np.zeros((nnodes,nnodes))
for edge in true_edges_igsp_paper:
    true_B_igsp_paper[edge] = 1
