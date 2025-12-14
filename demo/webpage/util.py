import time

from src.causalchange.causal_change import CausalChange
from src.causalchange.cc_types import DataMode, ScoreType, GraphSearch, GPType
from src.causalchange.gen.synthetic.gen_types import IvType, FunType, NoiseType
from src.causalchange.scoring.fit_cond_mixture import MixingType
import numpy as np

def run_cmm_on_node_pair(df1,df2,ret_model=False):
    mixed_X = np.vstack([ df1, df2 ]).T
    params = {}
    params["data_mode"] = DataMode.MIXED
    params["score_type"] = ScoreType.LIN
    params["mixing_type"] = MixingType.MIX_LIN
    top = CausalChange(**params)
    top.fit(mixed_X)
    dag = top.graph_state
    model = top


    model.visu_scatter_mixing_assignment_pair(0,1)
    model.visu_scatter_mixing_assignment_pair(1,0)
    if ret_model: return model


def load_colon_data():

    import rpy2.robjects as robjects
    robjects.r['load']("../datasets_prev/chapter6_mix/colon_data.RData")
    from rpy2.robjects import r, pandas2ri
    import pandas as pd

    #the data in this example is from https://www.biorxiv.org/content/10.1101/2020.08.02.233460v1.full
    colon_data = robjects.r['colon_data']

    data_dict = {name: pandas2ri.rpy2py(colon_data.rx2(name)) for name in colon_data.names}
    colon_df = pd.DataFrame({ky:vl for (ky, vl) in data_dict.items() if ky!="rnames"})

    print(colon_df.head())
    colon_df.to_csv("../datasets_prev/chapter6_mix/colon_data.csv", index=False)

    return colon_df

from src.causalchange.gen.generate import GenDataType, GSType, DagType, gen_data_type


def load_synthetic_data(seed, n_nodes):
    n_c = 10
    n_s_c = 1000
    params = {
        'N': n_nodes,
        'S': n_s_c * n_c,
        'P': 0.4,
        'C': n_c,
        'PC': 1,
        'Kmn': 1,
        'Kmx': 3,
        'IVM': GenDataType.MULTI_CONTEXT,
        'IVT': IvType.MIX,
        'GS': GSType.GRAPH,
        'DG': DagType.ERDOS,
        'F': FunType.MIX,
        'NS': NoiseType.GAUSS,
    }
    X, truths = gen_data_type(params, seed)

    return X, truths



def run_topic_on_data(X, ret_model=False):
    cc = CausalChange(
        data_mode=DataMode.CONTEXTS,
        graph_search=GraphSearch.TOPIC,
        score_type=ScoreType.GAM,
        vb=0,
    )

    G_hat = cc.fit(X)
    print(G_hat)
    if ret_model: return G_hat, cc
    return G_hat



def sanitize_matrix(mat):
    mat = np.array(mat, dtype=float)
    mat[~np.isfinite(mat)] = np.nan  # will become null in JSON
    return mat.tolist()


def sanitize_value(v):
    return float(v) if np.isfinite(v) else None
def replace_nans_with_none(obj):
    if isinstance(obj, dict):
        return {k: replace_nans_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nans_with_none(v) for v in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj
