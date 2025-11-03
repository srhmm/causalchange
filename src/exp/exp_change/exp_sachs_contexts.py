import os

import networkx as nx
import numpy as np
import pandas as pd

from src.causalchange.util.util import nxdigraph_to_lmg, compare_lmg_DAG
from src.exp.exp_change.algos import CD
from src.exp.exp_change.gen.sachs.sachs_utils import write_out_metrs


if __name__ == "__main__":
    files = {
        'cd3cd28.csv',
        'cd3cd28_icam2.csv',
        'cd3cd28_aktinhib.csv',
        'cd3cd28_g0076.csv',
        'cd3cd28_psitect.csv',
        'cd3cd28_u0126.csv',
        'cd3cd28_ly.csv',
        'pma.csv',
        'b2camp.csv',
    }
    dfs = {}
    fpath = "../dsets/dsets_context/sachs/Data/"
    for i, file in enumerate(files):
        varnms = pd.read_csv(fpath + file, delimiter=",", nrows=0).columns.tolist()
        df = pd.read_csv(fpath + file, delimiter=",", header=None, skiprows=1)
        #df.columns = cols
        dfs[i] = np.array(df)[0:400]

    gt = pd.read_csv("../dsets/dsets_context/sachs/GroundTruth.csv", delimiter=",")
    print(gt)
    name_to_idx = {name: i for i, name in enumerate(varnms)}
    edges = list(zip(
        gt['from'].map(name_to_idx),
        gt['to'].map(name_to_idx)
    ))
    edges = [(u, v) for u, v in edges if pd.notna(u) and pd.notna(v)]

    G = nx.DiGraph()
    G.add_nodes_from(range(len(varnms)))
    G.add_edges_from(edges)
    nx.set_node_attributes(G, {i: n for i, n in enumerate(varnms)}, name='label')
    cycles = list(nx.simple_cycles(G))  # each is a list of node IDs in order
    print(f"Found {len(cycles)} cycles")
    for c in cycles[:5]: print(c)
    labels = nx.get_node_attributes(G, 'label')
    def to_names(nodes): return [labels[i] for i in nodes]
    for c in cycles[:5]: print(" -> ".join(to_names(c + [c[0]])))  # pretty print closed loop
    edges_to_remove = []
    sccs = [s for s in nx.strongly_connected_components(G) if len(s) > 1]
    print("SCCs with cycles:", [to_names(sorted(s)) for s in sccs])
    for s in sccs:
        sub = G.subgraph(s).copy()
        e = nx.find_cycle(sub)[0][:2]
        edges_to_remove.append(e)

    G_acyclic = G.copy()
    G_acyclic.remove_edges_from(edges_to_remove)
    assert nx.is_directed_acyclic_graph(G_acyclic)

    #CD_ALGOS = [CD.TopicContextsGP] #[CD.TopicContextsRFF ]
    CD_ALGOS = [CD.TopicContextsRFF ]

    for cd_mthd in CD_ALGOS:
        print(f'Method: {cd_mthd.value}')
        cls = cd_mthd.get_method()
        kwargs = dict(vb=2)
        cls.fit(dfs, **kwargs)

        print(f"\tDiscovered Edges, {cd_mthd}:")
        for (i, j) in cls.dag.edges: print(f"\t\t{varnms[i]}->{varnms[j]}")
        #todo can check improvement matrix: evaluate_improvement(G_acyclic, improvement_matrix, higher_is_better=False)

        metrics = cls.get_graph_metrics(G_acyclic)
        out_fl = os.path.join("../../results/res_dixit", f"results_m_{cd_mthd}.tsv")
        os.makedirs(os.path.dirname(out_fl), exist_ok=True)
        write_out_metrs(out_fl, metrics)



        #out_fl = os.path.join("../../results_paper/res_sachs", f"results_m_{cd_mthd}.tsv")
        #os.makedirs(os.path.dirname(out_fl), exist_ok=True)
        #write_out_metrs(out_fl, metrics)
        if cd_mthd.value in [CD.TopicContextsGP.value, CD.TopicContextsRFF.value]:
            for node_i in cls.model.topic_graph.nodes:
                parents_i = list(cls.model.topic_graph.predecessors(node_i))
                score, res = cls.model._score(parents_i, node_i, ret_full_result=True)
                print(varnms[node_i], res["groups"])

        est_lmg = nxdigraph_to_lmg(cls.model.topic_graph)
        true_lmg = nxdigraph_to_lmg(G_acyclic)
        metrics = compare_lmg_DAG(true_lmg, est_lmg)


def evaluate_improvement(G, improvement, node_order=None, topn=10, higher_is_better=True):
    import numpy as np
    if node_order is None:
        node_order = list(range(improvement.shape[0]))
    idx = {n: i for i, n in enumerate(node_order)}
    n = len(node_order)
    adj = np.zeros((n, n), dtype=bool)
    for u, v in G.edges():
        if u in idx and v in idx:
            adj[idx[u], idx[v]] = True
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    y = adj[mask].astype(int)
    s = improvement[mask].astype(float)
    # Flip if lower is better
    if not higher_is_better:
        s = -s
    pos = s[y == 1];
    neg = s[y == 0]
    print(f"# Edges: {pos.size}, # Non-edges: {neg.size}")
    print(f"Mean score (edges):    {pos.mean():.4f} ± {pos.std():.4f}")
    print(f"Mean score (nonedges): {neg.mean():.4f} ± {neg.std():.4f}")
    # AUROC via ranks
    ranks = np.empty_like(s, dtype=float)
    order = np.argsort(s)
    ranks[order] = np.arange(1, s.size + 1, dtype=float)
    n_pos, n_neg = pos.size, neg.size
    if n_pos > 0 and n_neg > 0:
        auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        print(f"AUROC: {auc:.4f}")
    else:
        print("AUROC: N/A")
    k = max(1, n_pos)
    top_idx = np.argsort(-s)[:k]
    prec_at_k = y[top_idx].sum() / k
    print(f"Precision@{k} (k=#edges): {prec_at_k:.4f}")
    # Show top mistakes (now under the correct direction)
    ii, jj = np.where(mask)

    def pair(t):
        return (node_order[ii[t]], node_order[jj[t]])

    fp_idx = top_idx[y[top_idx] == 0][:topn]
    if fp_idx.size:
        print(f"\nTop {min(topn, fp_idx.size)} high-scoring NON-edges (FP):")
        for t in fp_idx:
            u, v = pair(t);
            print(f"  {u} -> {v}: score={(-s[t] if not higher_is_better else s[t]):.4f}")
    pos_all = np.where(y == 1)[0]
    if pos_all.size:
        worst_pos = pos_all[np.argsort(s[pos_all])[:topn]]
        print(f"\nBottom {min(topn, worst_pos.size)} low-scoring TRUE edges (FN):")
        for t in worst_pos:
            u, v = pair(t);
            print(f"  {u} -> {v}: score={(-s[t] if not higher_is_better else s[t]):.4f}")
