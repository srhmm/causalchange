#%%
from causalchange.gen.generate import GenDataType, GSType, DagType

import networkx as nx
import pandas as pd

from pathlib import Path
from causalchange.gen.generate import gen_data_type
from causalchange.gen.synthetic.gen_types import IvType, FunType, NoiseType

from causalchange.old.causal_change_large import CausalChange

#sys.path.insert(0, os.path.abspath(".."))
# or sys.path.insert(0, os.path.abspath("/workspace"))
#%%

seed = 42

n_c = 10
n_s_c = 500
for seed in range(10):
    params = {
        'N': 5,
        'S': n_s_c*n_c,
        'P': 0.4,
        'C': n_c,
        'PC': 1,
        'Kmn': 1,
        'Kmx': 3,
        'IVM': GenDataType.MULTI_CONTEXT,
        'IVT': IvType.HARD,
        #'GS': GSType.BIV_CAUSAL_CHANGEX,
         'GS': GSType.GRAPH,
        'DG': DagType.ERDOS,
        'F': FunType.LIN,
        'NS': NoiseType.GAUSS,
    }

    X, truths = gen_data_type(params, seed)
    true_g = truths["true_g"]
    Path("../demo/datasets/").mkdir(parents=True, exist_ok=True)
    for c_i in range(len(X)):
         pd.DataFrame(X[c_i]).to_csv(f'../demo/datasets/synthetic_IID.tsv',  sep='\t', index=False)

    pd.DataFrame(nx.to_numpy_array(truths["true_g"])).to_csv('../demo/datasets/synthetic_IID_g.tsv',
                                                             sep='\t', index=False)

    #%%
    #truths['_dg'].plot_X()
    #%%
    #truths['true_g'].edges
    #%%
    #truths['_dg'].plot_conditionals_under(truths['true_g'])
    #%%
    from causalchange._cc_types import DataMode, ScoreType, GraphSearch

    for score_type in [ ScoreType.KRR ]:
        cc = CausalChange(
            data_mode=DataMode.CONTEXTS,
            score_type=score_type,
            graph_search=GraphSearch.CHAIN, vb=3,
            truths=truths)
        dag = cc.fit(X)
        #%%
        print(cc.topological_order)
        print(dag.edges)
        print(list( nx.topological_sort(truths['true_g'])))
        print(true_g.edges)
