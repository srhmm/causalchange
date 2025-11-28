#%% md
# # Case Study: Causal Discovery (i.i.d. data)
# 
# Figure 5
#%% md
# Directory to store the downloaded causal chamber datasets.
#%%
DOWNLOAD_DIR = ''
#%%
import sys
sys.path = ['../'] + sys.path
#%%
import pandas as pd
import numpy as np
import sempler.plot
import matplotlib
import matplotlib.pyplot as plt
#%% md
# ## Auxiliary Code
#%% md
# #### Plot settings
#%%
matplotlib.rc('text', usetex = True)
matplotlib.rc('font', **{'family' : "sans-serif"})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

matplotlib.rcParams.update({'font.size': 7})
MM = 1/25.4 # inches to mm

def to_rgb(H, b=1, a=1):
    RGBa = []
    for h in H:
        h = h.lstrip("#")
        RGBa.append(tuple(int(h[i:i+2], 16) / 256 * b for i in (0, 2, 4)) + (a,))
    return np.array(RGBa)

# Color palettes
color_blind_1 = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4', '#020202',]
color_blind_2 = ['#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac']
color_blind_3 = ['#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1', '#35978f', '#01665e']
ibm_color_blind = ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000", "#000000", "#ffffff"]
rainbow10 = ["#ff4365", "#ffdd43", "#59ff43", "#43ffdd", "#7395ff", "#4365ff", "#e943ff", "#601e9e", "#6a6a6a", "#964b00"]


# Pick one and show

palette = ibm_color_blind

base = to_rgb(palette)
basea = to_rgb(palette, a=0.5)
baseb = to_rgb(palette, b=0.7)
plt.scatter(np.arange(len(base)), np.zeros(len(base)), c = base)
plt.scatter(np.arange(len(base)), np.ones(len(base))*0.5, c = basea)
plt.scatter(np.arange(len(base)), np.ones(len(base))*-0.5, c = baseb)
#%% md
# ### Metrics for ground-truth effect recovery
# 
# Precision: $\frac{TP}{TP+FP}\quad\quad$ Recall: $\frac{TP}{P}$
#%%
def precision(estimate, truth):
    # TP = edges in estimate that are also in truth
    # FP + TP = total edges in estimate
    return np.logical_and(estimate,truth).sum() / estimate.sum()

def recall(estimate, truth):
    # TP = edges in estimate that are also in truth
    # P = total edges in truth
    return np.logical_and(estimate,truth).sum() / truth.sum()
#%% md
# ## Define ground-truth graph
#%%
import causalchamber
from causalchamber.utils import graph_to_tikz
from causalchamber.ground_truth import latex_name
#%% md
# For standard configuration of the light tunnel we can simply take the subgraph induced by the variables we use (this is not the case for e.g. linked configuration)
#%%
variables = ['red', 'green', 'blue', 'current', 'ir_1', 'ir_2', 'ir_3', 'vis_1', 'vis_2', 'vis_3', 'pol_1', 'pol_2', 'angle_1', 'angle_2', 'l_11', 'l_12', 'l_21', 'l_22', 'l_31', 'l_32']
#%%
# Ground truth graph
true_dag = causalchamber.ground_truth.graph('lt', 'standard').loc[variables, variables].values
sempler.plot.plot_graph(true_dag, labels = variables)
print(graph_to_tikz(true_dag, radius=1.7, labels=[latex_name(v) for v in variables]))
#%% md
# ## Download dataset
#%%
from causalchamber.datasets import Dataset
dataset = Dataset(name="lt_interventions_standard_v1", root='')
#%%
experiments = [
    "uniform_reference",
    "uniform_red_strong",
    "uniform_green_strong",
    "uniform_blue_strong",
    "uniform_v_c_strong",
    "uniform_t_ir_1_strong",
    "uniform_t_ir_2_strong",
    "uniform_t_ir_3_strong",
    "uniform_t_vis_1_strong",
    "uniform_t_vis_2_strong",
    "uniform_t_vis_3_strong",
    "uniform_pol_1_strong",
    "uniform_pol_2_strong",
    "uniform_v_angle_1_strong",
    "uniform_v_angle_2_strong",
    "uniform_l_11_mid",
    "uniform_l_12_mid",
    "uniform_l_21_mid",
    "uniform_l_22_mid",
    "uniform_l_31_mid",
    "uniform_l_32_mid",
    
]
observational_data = dataset.get_experiment(experiments[0]).as_pandas_dataframe()[variables].values
interventional_data = [dataset.get_experiment(e).as_pandas_dataframe()[variables].values for e in experiments]
#%%
print("Sample sizes:")
for e,df in zip(experiments,interventional_data):
    print(f"  {len(df):<5}  {e}")
#%% md
# ## Task a1: Observational data

#%%
sub_sample = 500
sub_exp = 5
interventional_data_dict = {i: dataset.get_experiment(e).as_pandas_dataframe()[variables].sample(n=sub_sample, replace=False, random_state=42).values for i,e in enumerate(experiments[:sub_exp])}

from src.causalchange.causal_change import CausalChange
#%%
import networkx as nx
print([variables[i] for i in nx.topological_sort(nx.from_numpy_array(true_dag, create_using=nx.DiGraph))])
#%%

', '.join([f'{variables[i] }->{variables[j] }' for i, j in  nx.from_numpy_array(true_dag, create_using=nx.DiGraph).edges ])
#%%
from src.causalchange.cc_types import DataMode, ScoreType, GraphSearch

cc = CausalChange(
    data_mode=DataMode.CONTEXTS,
    score_type=ScoreType.KRR,
    graph_search=GraphSearch.CHAIN,
    node_nms=variables,
    vb=3 )
dag = cc.fit(interventional_data_dict)

#%%
est_dag_nx = dag
dg = nx.to_numpy_array(dag)
true_nx = nx.from_numpy_array(true_dag, create_using=nx.DiGraph)
print(precision(dg, true_dag))
print(f'Precision: {precision(dg, true_dag)}')
print(f'Recall: {recall(dg, true_dag)}')
print(f'True Order : {[i for i in nx.topological_sort(nx.from_numpy_array(true_dag, create_using=nx.DiGraph))]}')
print(f'Est Order : {[i for i in nx.topological_sort(dag)]}')
#est_order =  [14, 16, 17, 15, 10, 19, 11, 18, 3, 1, 13, 0, 12, 2, 6, 5, 4, 9, 8, 7]
for i in nx.topological_sort(dag):
    print(f'{variables[i]}: pre={len(list(true_nx.predecessors(i)))}, succ={len(list(true_nx.successors(i)))}')
#%%
