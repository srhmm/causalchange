from src.coco.util.utils import f1_score, tpr, fpr


def eval_confounded(sim_01, nodes_confounded):
    ''' Evaluates confounding for each node pair.

    :param sim_01: (n_x, n_x), confounded decision
    :param nodes_confounded: [[X_i for each X_i confounded by Z_j] for each latent Z_j]
    :return:
    '''
    tp, fp, tn, fn = 0, 0, 0, 0
    tfp, ttp, ffp = 0, 0, 0

    for node_i in sim_01:
        for node_j in sim_01[node_i]:
            #if skip_source_nodes and \
            #        (len([i for i in G.predecessors(node_i)] )==0 or len([i for i in G.predecessors(node_j)] )==0):
            #    continue
            cf = True in [(node_i in lst) and (node_j in lst) for lst in nodes_confounded]

            # TODO for debug, can remove from here on:
            if sim_01[node_i][node_j]:
                if cf:
                    tp += 1
                else:
                    fp += 1
            else:
                if cf:
                    fn += 1
                else:
                    tn += 1

    return tp, fp, tn, fn, \
           f1_score(tp, fp, fn), \
           tpr(tp, fn), fpr(fp, tn)

def eval_causal_confounded(G_estim, dag, sim_01, nodes_confounded):
    ''' Jointly evaluates causal decisions and confounded decisons

    :param G_estim: estimated DAG, usually the result of coco._discover_DAG() which uses the MSS score (Perry et al. 2022)
    :param dag: true DAG
    :param sim_01: for CoCo, (n_X, n_X) confounded decisions for node pairs; for MSS, empty.
    :param nodes_confounded: true confounded nodes, [[X_i for each X_i confounded by Z_j] for each latent Z_j]
    :return:
    '''
    G_obs = dag.G
    #causal
    tp, fp, tn, fn = 0, 0, 0, 0
    #causal vs. confounded
    tp_tp, tp_fp, tp_tn, tp_fn = 0, 0, 0, 0
    fp_tp, fp_fp, fp_tn, fp_fn = 0, 0, 0, 0
    tn_tp, tn_fp, tn_tn, tn_fn = 0, 0, 0, 0
    fn_tp, fn_fp, fn_tn, fn_fn = 0, 0, 0, 0

    for node_i in G_obs.nodes:
        for node_j in G_obs.nodes:
            if node_i == node_j:
                continue
            causal = ((node_i, node_j) in G_obs.edges)
            confounded = True in [(node_i in lst) and (node_j in lst) for lst in nodes_confounded]
            estim_causal = ((node_i, node_j) in G_estim.edges)
            if node_j in sim_01[node_i]:
                estim_confounded = (sim_01[node_i][node_j] == 1)
            else:
                estim_confounded = (sim_01[node_j][node_i] == 1)

            if estim_causal:
                if causal:
                    tp += 1
                    if estim_confounded:
                        if confounded:
                            tp_tp += 1
                        else:
                            tp_fp += 1
                    else:
                        if confounded:
                            tp_fn += 1
                        else:
                            tp_tn += 1
                else:
                    fp += 1
                    if estim_confounded:
                        if confounded:
                            fp_tp += 1
                        else:
                            fp_fp += 1
                    else:
                        if confounded:
                            fp_fn += 1
                        else:
                            fp_tn += 1
            else:
                if causal:
                    fn += 1
                    if estim_confounded:
                        if confounded:
                            fn_tp += 1
                        else:
                            fn_fp += 1
                    else:
                        if confounded:
                            fn_fn += 1
                        else:
                            fn_tn += 1
                else:
                    tn += 1
                    if estim_confounded:
                        if confounded:
                            tn_tp += 1
                        else:
                            tn_fp += 1
                    else:
                        if confounded:
                            tn_fn += 1
                        else:
                            tn_tn += 1
    return tp, fp, tn, fn, \
           f1_score(tp, fp, fn), \
           tpr(tp, fn), fpr(fp, tn), \
           tp_tp, tp_fp, tp_tn, tp_fn,\
           fp_tp, fp_fp, fp_tn, fp_fn,\
           tn_tp, tn_fp, tn_tn, tn_fn,\
           fn_tp, fn_fp, fn_tn, fn_fn,

