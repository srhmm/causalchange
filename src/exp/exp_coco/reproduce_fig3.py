import networkx as nx
import numpy as np
import pandas as pd

from src.coco.coco import CoCo
from src.coco.util.co_test_types import CoShiftTestType, CoCoTestType, CoDAGType
from src.coco.util.mi_sampling import Sampler


def show_raf_mek(path):
    node_raf, node_mek, node_pkc = 0, 1, 8
    show_sachs_nodes(0, 1, path)


def show_sachs_nodes(node_x, node_y, nms, path="experiments/data_cytometry"):
    Dc = [
        np.log(
            pd.read_csv(f'{path}/dataset_{i}.csv')
        ) for i in range(1, 10)
    ]
    nms = Dc[0].columns
    Dc = [np.array(X[:707]) for X in Dc]
    Dc = np.array(Dc)
    import matplotlib.pyplot as plt

    n_c = len(Dc)
    X = Dc[:, :, node_x]
    y = Dc[:, :, node_y]
    # for n_context in range(n_c):
    #    plt.scatter(X[n_context], y[n_context], label='Context ' + str(n_context))

    observ_context, interv_context = 0, 5
    plt.scatter(X[observ_context], y[observ_context], label='Observational Context ' + str(observ_context))
    plt.scatter(X[interv_context], y[interv_context], label='Interventional Context ' + str(interv_context))

    write_file = open(f"{path}/tex_sachs_{nms[node_x]}_{nms[node_y]}.csv", "a+")
    write_file.write("X_o\ty_o\tX_i\ty_i")
    for i in range(len(X[observ_context])):
        write_file.write(
            f'\n{X[observ_context][i]}\t{y[observ_context][i]}\t{X[interv_context][i]}\t{y[interv_context][i]}')
    write_file.close()
    plt.legend()
    plt.xlabel(str(nms[node_x]))
    plt.ylabel(str(nms[node_y]))
    plt.title('Edge ' + str(nms[node_x]) + ' -> ' + str(nms[node_y]))


def reproduce_fig3(path="/data_cytometry"):
    # standard params for coco
    SHIFT_TEST = CoShiftTestType.PI_KCI
    CONFOUNDING_TEST = CoCoTestType.MI_ZTEST
    DAG_SEARCH = CoDAGType.SKIP
    ALPHA_SHIFT_TEST = 0.05

    # true_cpdag = dag2cpdag(true_dag)
    # mec_size = len(cpdag2dags(true_cpdag))
    # total_edges = np.sum(true_dag)
    # unoriented_edges = np.sum((true_cpdag + true_cpdag.T) == 2) // 2

    # DAG as in nocadillac

    dag = np.zeros((11, 11))

    dag[8, np.asarray([10, 7, 0, 9])] = 1
    dag[2, np.asarray([3, 7])] = 1
    dag[7, 5] = 1
    dag[0, 1] = 1
    dag[3, 8] = 1
    dag[4, np.asarray([10, 7, 9, 1, 6])] = 1
    dag[1, 5] = 1
    dag[5, np.asarray([6, 3])] = 1

    nodes_true = range(len(dag))
    results = [0 for _ in nodes_true]

    Dc = [
        np.log(
            pd.read_csv(f'{path}/dataset_{i}.csv')
        ) for i in range(1, 10)
    ]

    nms = Dc[0].columns
    Dc = [np.array(X[:707]) for X in Dc]
    Dc = np.array(Dc)
    s_result_fci_jci = "FCI_JCI\n"
    s_result_coco = "COCO\n"

    for confounder_i, confounder in [(8, 8)]:  # enumerate(nodes_true):
        nodes = [n for n in nodes_true if n != confounder]
        G = nx.DiGraph([])
        edges = []
        G_observed = nx.DiGraph([])
        edges_observed = []
        print("\nCONFOUNDER:", nms[confounder])
        s_result_fci_jci += "\nCONFOUNDER:" + nms[confounder]
        s_result_coco += "\nCONFOUNDER:" + nms[confounder]
        for ind, i in enumerate(nodes_true):
            for j in nodes_true:
                if dag[i][j] == 1:
                    edges.append((i, j))
                    if i != confounder:
                        edges_observed.append((i, j))
                        s_result_coco += f"\nObserved\t{i}.{nms[i]}->{j}.{nms[j]}"
                    else:
                        s_result_coco += f"\nConfounded\t{i}.{nms[i]}->{j}.{nms[j]}"

        s_result_coco += "\n\n"
        G.add_nodes_from(nodes_true)
        G.add_edges_from(edges)
        G_observed.add_nodes_from(nodes)
        G_observed.add_edges_from(edges_observed)
        nms_observed = [nm for i, nm in enumerate(nms) if i != confounder]
        Dobs = np.array([Dc[ci][:, nodes] for ci in range(len(Dc))])

        class cls:
            def __init__(self, G):
                self.G = G
        dg = cls(G)
        coco = CoCo(Dc, G_observed, Sampler(), CONFOUNDING_TEST, SHIFT_TEST, DAG_SEARCH,
                    n_components=1, dag=dg, node_nms=nms, alpha_shift_test=ALPHA_SHIFT_TEST)

        # already done in coco
        coco._estimated_graph_cuts_n(1)
        results[confounder_i] = coco

        print("")
        nm = nms
        if (len(coco.maps_estimated) == 10):
            nm = nms_observed
        if len(coco.estimated_cuts):
            print("\tCUTS:", coco.estimated_cuts, [nm[j] for j in coco.estimated_cuts[0]])
            s_result_coco += f"\n{coco.estimated_cuts}" + str([nm[j] for j in coco.estimated_cuts[0]])
        for (i, j) in coco.G_.edges():
            print(f"{nms[i]}->{nms[j]}")
        # for n_i, n_j in coco.G_estimated.edges():
        #    print(f"{nms[n_i]}->{nms[n_j]}")# -> provide the true DAG.

        # fci = FCI_JCI(Dobs, G_observed, G, None,   independence_test='fisherz',
        #                                                          method=MethodType.FCI_JCI)
        # for i in range(len(fcii.G_estim_bg.graph)):
        #    for j in range(len(fcii.G_estim_bg.graph)):
        #        if fci.G_estim_bg.graph[i][j] == 1 and fci.G_estim_bg.graph[j][i] == 1:
        #            print(i, j, nms_observed[i], nms_observed[j])
        #            s_result_fci_jci += f"\n\t{i}. {nms_observed[i]} <-> {j}. {nms_observed[j]} by {nms[confounder]}"

        print()
        # praf, pmek, pip3, p38, pka, jnk, pkc

        for i in range(len(results)):
            confounder = i
            print(f"CFD: {i}. {nms[i]}")
            nms_observed = [nm for i, nm in enumerate(nms) if i != confounder]
            nm = nms
            if (len(results[i].maps_estimated) == 10):
                nm = nms_observed
                print("\tVARS:", nms_observed)
            else:
                print("\t", [v for v in nms])
            results[i]._estimated_graph_cuts_n(1)
            if len(results[i].estimated_cuts):
                print("\tCUTS:", results[i].estimated_cuts,
                      [nm[j] for j in
                       results[i].estimated_cuts[0]])  # , nms[[j for j in results[i].estimated_cuts][0]])

    # CPDAG in sparse shift
    dag_nocadillac_paper = False
    dag_mss_paper = True
    if dag_mss_paper:
        dag[2, np.asarray([3, 4])] = 1
        dag[4, 3] = 1
        dag[8, np.asarray([10, 7, 0, 1, 9])] = 1
        dag[7, np.asarray([0, 1, 5, 6, 9, 10])] = 1
        dag[0, 1] = 1
        dag[1, 5] = 1
        dag[5, 6] = 1
