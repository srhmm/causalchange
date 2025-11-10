import os

import networkx as nx

from src.causalchange.util.util import nxdigraph_to_lmg, compare_lmg_DAG
from src.exp.exp_change.algos import CD
from src.exp.exp_change.dixit.dixit_meta import dixit_get_samples, true_B_dixit_paper
from src.exp.exp_change.gen.sachs.sachs_utils import (write_out_metrs)

if __name__ == "__main__":

    obs_samples, setting_list = dixit_get_samples()

    X = dict()
    X[0] = obs_samples
    for i, dic in enumerate(setting_list):
        X[i+1] = dic['samples'][ 0:500, :]
    true_B =     true_B_dixit_paper #igsp
    true_nx = nx.from_numpy_array(true_B.T)
    true_lmg = nxdigraph_to_lmg(true_nx)


    CD_ALGOS = [CD.TopicContextsRFF] #[CD.TopicContextsRFF ]

    for cd_mthd in CD_ALGOS:
        print(f'Method: {cd_mthd.value}')
        cls = cd_mthd.get_method()
        kwargs = dict(vb=2)
        cls.fit(X, **kwargs)

        print(f"\tDiscovered Edges, {cd_mthd}:")
        for (i, j) in cls.dag.edges: print(f"\t\t{i}->{j}")
        #todo to check improvement matrix: evaluate_improvement(G_acyclic, improvement_matrix, higher_is_better=False)

        metrics = cls.get_graph_metrics(true_nx)

        out_fl = os.path.join("../../results/res_dixit", f"results_m_{cd_mthd}.tsv")
        os.makedirs(os.path.dirname(out_fl), exist_ok=True)
        write_out_metrs(out_fl, metrics)


        if cd_mthd.value in [CD.TopicContextsGP.value, CD.TopicContextsRFF.value]:
            for node_i in cls.model.graph_state.nodes:
                parents_i = list(cls.model.graph_state.predecessors(node_i))
                score, res = cls.model._score(parents_i, node_i, ret_full_result=True)
                print(node_i, res["groups"])

        est_lmg = nxdigraph_to_lmg(cls.model.graph_state)
        metrics = compare_lmg_DAG(true_lmg, est_lmg)
        print(metrics)

