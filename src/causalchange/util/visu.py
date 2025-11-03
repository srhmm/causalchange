import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import expected_mutual_information
from sklearn.mixture import GaussianMixture
from src.causalchange.util.utils_idl import exp_mutual_info_score, soft_mutual_information, expected_smi, \
    soft_adjusted_mutual_information, pi_join


def visu_kl_sample_heatmaps(kl_sample_matrices, node_name="unknown", p_proba=None, svfl=None,
                                vmin=None, vmax=None, cmap="YlOrRd", #"magma",
                            sort="posterior"):
    """
    Plots KL divergence heatmaps for each pair of mixture components.

    :param kl_sample_matrices: dict[(k,l)] -> ndarray [N_samples, N_samples] of KL divergences.
    :param node_name: Name of the node for the plot title.
    :param p_proba: Optional. Numpy array of shape [N_samples, K] for posterior probabilities.
    :param svfl: Path to save plots.
    :param vmin: Color scale lower limit.
    :param vmax: Color scale upper limit.
    :param cmap: Colormap.
    :param sort: Sorting strategy. 'kl' for average KL, 'posterior' for cluster confidence, None for raw order.
    """
    os.makedirs(svfl, exist_ok=True) if svfl is not None else None

    for (k, l), kl_mat in kl_sample_matrices.items():
        if sort == "kl":
            order_k = np.argsort(np.mean(kl_mat, axis=1))
            order_l = np.argsort(np.mean(kl_mat, axis=0))
        elif sort == "posterior":
            assert p_proba is not None
            order_k = np.argsort(-p_proba[:, k])  # sort desc: high p(k) first
            order_l = np.argsort(-p_proba[:, l])
        else:
            order_k = np.arange(kl_mat.shape[0])
            order_l = np.arange(kl_mat.shape[1])

        kl_mat_sorted = kl_mat[np.ix_(order_k, order_l)]

        # Plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(kl_mat_sorted, cmap=cmap, vmin=vmin, vmax=vmax)
        sort_label = f" (sorted by {sort})" if sort else ""
        plt.title(f"Node {node_name}: KL Div {k} → {l}{sort_label}")
        plt.xlabel(f"samples assigned to {l}")
        plt.ylabel(f"samples assigned to {k}")
        plt.tight_layout()

        if svfl is not None:
            fname = f"{svfl}/node_{node_name}_samp_sep_k{k}_l{l}_{sort}.png"
            plt.savefig(fname)
            plt.close()
        else:
            plt.show()


def visu_component_heatmaps(kl_component_mats, skip_singleclus=False, svfl=None, method_idf='', figsz=(4, 4)):
    for node_i, div_matrix in kl_component_mats.items():
        n_comp = div_matrix.shape[0]
        if n_comp == 0 and skip_singleclus: continue
        plt.figure(figsize=figsz)
        sns.heatmap(div_matrix, annot=True, cmap="YlOrRd", square=True,
                    cbar_kws={'label': 'KL div'},
                    xticklabels=[f"C{c}" for c in range(n_comp)],
                    yticklabels=[f"C{c}" for c in range(n_comp)])

        plt.title(f"Component Separation (Node {node_i})")
        plt.tight_layout()

        if svfl is not None:
            os.makedirs(svfl, exist_ok=True)
            plt.savefig(os.path.join(svfl, f"node_{node_i}_cmpnt_sep_{method_idf}.png"))
            plt.close()
        else:
            plt.show()

def visu_node_idls(G, X, idls, svfl=None, method_idf='_estimZ', figsize=(7, 5)):
    """ Plots node mixing labels """

    for i in G:
        pa_i = list(G.predecessors(i))
        fig, axes = plt.subplots(len(pa_i) + 1, 1, figsize=figsize)

        labels = idls[i]
        ax1 = axes if len(pa_i) < 1 else axes[0]
        ax1.set_title(f"Node {i}: P(X{i})")

        cmap_i = plt.cm.get_cmap('tab10', len(np.unique(labels)))
        ax1.scatter(
            np.random.normal(size=X[:, i].shape),
            X[:, i], c=labels,
            cmap=cmap_i
        )
        for ix, pa in enumerate(pa_i):
            axes[1 + ix].scatter(X[:, pa], X[:, i], c=labels, cmap='viridis')
            axes[1 + ix].set_title(f"Causal Relationship: P(X{i} | X{pa}) ")

        plt.tight_layout()

        plt.tight_layout()
        if svfl is not None:
            os.makedirs(svfl, exist_ok=True)
            plt.savefig(svfl + f"node_{i}_mixing_pa_{pa_i}{method_idf}")
            plt.close()
        else:
            plt.show()

def visu_node_idl(X, i, pa_i, labels, svfl=None, method_idf='_estimated', figsize=(7,5)):
    fig, axes = plt.subplots(len(pa_i) + 1, 1, figsize=figsize)

    ax1 = axes if len(pa_i) < 1 else axes[0]
    ax1.set_title(f"Node {i}: P(X{i})")

    cmap_i = plt.cm.get_cmap('tab10', len(np.unique(labels)))
    ax1.scatter(
        np.random.normal(size=X[:, i].shape),
        X[:, i], c=labels,
        cmap=cmap_i
    )
    for ix, pa in enumerate(pa_i):
        axes[1 + ix].scatter(X[:, pa], X[:, i], c=labels, cmap='viridis')
        axes[1 + ix].set_title(f"Causal Relationship: P(X{i} | X{pa}) ")

    plt.tight_layout()

    plt.tight_layout()
    if svfl is not None:
        os.makedirs(svfl, exist_ok=True)
        plt.savefig(svfl + f"node_{i}_pa_{pa_i}{method_idf}")
        plt.close()
    else:
        plt.show()

def visu_node_true_idl(X, i, pa_i, t_n_Zs, t_Zs, svfl=None, method_idf='_true', figsize=(7,5)):
    fig, axes = plt.subplots(len(pa_i) + 1, 1, figsize=figsize)

    ax1 = axes if len(pa_i) < 1 else axes[0]
    ax1.set_title(f"TRUTH: Node {i}: P(X{i})")
    true_labels = np.zeros(X.shape[0])
    for zi, node_set in enumerate(t_n_Zs):
        if i in node_set:
            true_labels = pi_join(true_labels, t_Zs[zi])

    cmap_i = plt.cm.get_cmap('tab10', len(np.unique(true_labels)))
    ax1.scatter(
        np.random.normal(size=X[:, i].shape),
        X[:, i], c=true_labels,
        cmap=cmap_i
    )
    for ix, pa in enumerate(pa_i):
        axes[1 + ix].scatter(X[:, pa], X[:, i], c=true_labels, cmap='viridis')
        axes[1 + ix].set_title(f"Causal Relationship: P(X{i} | X{pa}) ")

    plt.tight_layout()

    plt.tight_layout()
    if svfl is not None:
        os.makedirs(svfl, exist_ok=True)
        plt.savefig(svfl + f"node_{i}_pa_{pa_i}{method_idf}")
        plt.close()
    else:
        plt.show()


def visu_pproba_dens(pproba):
    import seaborn as sns
    n_components = pproba.shape[1]
    plt.figure(figsize=(8, 4))

    for k in range(n_components):
        sns.kdeplot(pproba[:, k], label=f'Component {k}', fill=True, alpha=0.3)

    plt.xlabel('Post. Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

def visu_node_pproba(G, X, idls, pproba_dict, svfl=None, method_idf='', figsize=(7, 5)):
    """ Plots node soft mixing labels """
    for i in G:
        pa_i = list(G.predecessors(i))
        fig, axes = plt.subplots(len(pa_i) + 1, 1, figsize=figsize)
        labels = idls[i]

        confidences = np.max(pproba_dict[i], axis=1)

        ax1 = axes[0] if len(pa_i) > 0 else axes
        ax1.set_title(f"Node {i}: P(X{i})")
        cmap_i = plt.cm.get_cmap('tab10', len(np.unique(labels)))

        sc1 = ax1.scatter(
            np.random.normal(size=X[:, i].shape), X[:, i], c=labels,
            cmap=cmap_i, label='Class Labels'
        )
        fig.colorbar(sc1, ax=ax1, label="Class Labels")
        for ix, pa in enumerate(pa_i):
            ax2 = axes[1 + ix]
            ax2.set_title(f"Causal Relationship: P(X{i} | X{pa})")

            sc2 = ax2.scatter(
                X[:, pa], X[:, i], c=confidences, cmap='viridis', alpha=0.8
            )
            fig.colorbar(sc2, ax=ax2, label="Confidence Level")

        plt.tight_layout()

        if svfl is not None:
            os.makedirs(svfl, exist_ok=True)
            plt.savefig(svfl + f"Node_{i}_pa_{pa_i}_estim_{method_idf}")
            plt.close()
        else: plt.show()

def visu_node_conf_histograms(node_pproba, ncols=3, bins=20):
    """ Plots confidence histograms   """
    import matplotlib.pyplot as plt
    import numpy as np
    models = list(node_pproba.keys())
    proba_list = list(node_pproba.values())

    n = len(proba_list)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    plt.figure(figsize=(4 * ncols, 3.5 * nrows))
    for i, (model_name, p) in enumerate(zip(models, proba_list)):
        confidences = np.max(p, axis=1)
        avg_confidence = np.mean(confidences)

        plt.subplot(nrows, ncols, i + 1)
        plt.hist(confidences, bins=bins, range=(0, 1), color='mediumslateblue', alpha=0.7)
        plt.title(f"{model_name}\navg conf: {avg_confidence:.3f}")
        plt.ylabel("ct")

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import os


def visu_pair_ovl(G, X, idls, node_pproba, svfl=None, method_idf='', figsize=(30,30)):
    """  pairwise marg distributions of nodes, cluster assignments, overlap."""
    node_names = list(G.nodes)
    n_nodes = len(node_names)
    fig, axes = plt.subplots(n_nodes, n_nodes, figsize=figsize)

    for i, node1 in enumerate(node_names):
        for j, node2 in enumerate(node_names):
            ax = axes[i, j]
            if i == j:
                # diagonal
                labels = idls[node1]
                cmap = plt.cm.get_cmap('tab10', len(np.unique(labels)))
                ax.scatter(
                    np.random.normal(size=X[:, node1].shape),
                    X[:, node1], c=labels, cmap=cmap, alpha=0.7
                )
                ax.set_title(f"node {node1}")
                ax.set_xlabel(f"X{node1}")
            else:
                # off-diagonal
                p1 = node_pproba[node1]
                p2 = node_pproba[node2]
                cluster1 = np.argmax(p1, axis=1)  # Most probable cluster for node1
                cluster2 = np.argmax(p2, axis=1)  # Most probable cluster for node2

                ax.scatter(
                    X[:, node1], X[:, node2], c=cluster1, cmap='viridis', alpha=0.7, label=f"assignment (X{node1})"
                )
                ax.scatter(
                    X[:, node1], X[:, node2], c=cluster2, cmap='coolwarm', alpha=0.5,
                    label=f"assignment (X{node2})"
                )
                ax.set_title(f"pair X{node1} vs X{node2}")
                ax.set_xlabel(f"X{node1}")
                ax.set_ylabel(f"X{node2}")

                # Show the overlap region using alpha blending
                ax.legend(loc='best', fontsize=8)

            plt.tight_layout()

    # Save or show the plot
    if svfl is not None:
        os.makedirs(svfl, exist_ok=True)
        plt.savefig(svfl + f"visu_pairwise_assignments_{method_idf}.png")
        plt.close()
    else:
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics import adjusted_mutual_info_score


def visu_pair_mi(node_idl, node_pprobas, soft=False, svfl=None, method_idf='',
                 figsz=(4, 4), hide_diag=False, hide_singleclus=False,
                 hide_insig=False, annot=False, cmap="YlGnBu", use_clustermap=True):
    node_names = list(node_idl.keys())
    n_nodes = len(node_names)

    smi_matrix = np.zeros((n_nodes, n_nodes))

    for i, node1 in enumerate(node_names):
        for j, node2 in enumerate(node_names):

            if i == j and hide_diag:
                continue

            if i != j and hide_singleclus and len(np.unique(node_idl[node1])) == 1 and len(
                    np.unique(node_idl[node2])) == 1:
                continue

            if soft:
                smi_matrix[i, j] = soft_mutual_information(node_pprobas[node1], node_pprobas[node2])
            else:
                smi_matrix[i, j] = adjusted_mutual_info_score(node_idl[node1], node_idl[node2])

            if hide_insig:
                if soft and smi_matrix[i, j] < \
                        expected_smi(node_pprobas[node1], node_pprobas[node2], len(node_idl[node1]))[0]:
                    smi_matrix[i, j] = 0
                if not soft and smi_matrix[i, j] < exp_mutual_info_score(node_idl[node1], node_idl[node2])[0]:
                    smi_matrix[i, j] = 0

    # Perform hierarchical clustering to reorder nodes
    distance_matrix = 1 - smi_matrix
    np.fill_diagonal(distance_matrix, 0)  # Ensure self-distance is zero

    # Avoid issues if all distances are identical (e.g. all zeros)
    if np.any(distance_matrix[np.triu_indices(n_nodes, k=1)] > 0):
        condensed_distance = distance_matrix[np.triu_indices(n_nodes, k=1)]
        Z = linkage(condensed_distance, method='average')
        ordered_indices = leaves_list(Z)
    else:
        ordered_indices = np.arange(n_nodes)

    # Apply the ordering
    smi_matrix = smi_matrix[np.ix_(ordered_indices, ordered_indices)]
    ordered_names = [node_names[i] for i in ordered_indices]

    import seaborn as sns
    # Plotting
    if use_clustermap:

        # Use seaborn clustermap
        cg = sns.clustermap(smi_matrix, cmap=cmap, row_cluster=False, col_cluster=False,
                            xticklabels=ordered_names, yticklabels=ordered_names,
                            figsize=figsz, annot=annot)
        cg.ax_heatmap.set_title("SMI pairwise (soft)" if soft else "AMI pairwise")
    else:
        plt.figure(figsize=figsz)
        sns.heatmap(smi_matrix, annot=annot, cmap=cmap,
                    xticklabels=ordered_names, yticklabels=ordered_names)
        plt.title("SMI pairwise (soft)" if soft else "AMI pairwise")
        plt.tight_layout()

        if svfl is not None:
            os.makedirs(svfl, exist_ok=True)
            plt.savefig(os.path.join(svfl, f"visu_pair_smi_{method_idf}.png"))
            plt.close()
            return
        else:
            plt.show()
            return

    # Save or show the clustermap
    if svfl is not None:
        os.makedirs(svfl, exist_ok=True)
        cg.savefig(os.path.join(svfl, f"visu_pair_smi_{method_idf}.png"))
        plt.close()
    else:
        plt.show()
    return smi_matrix

def visu_pair_mi_unordered(node_idl, node_pprobas, soft=False, svfl=None, method_idf='', figsz=(4,4), hide_diag=False, hide_singleclus=False, hide_insig=False, annot=False, cmap="YlGnBu"):
    node_names = list(node_idl.keys())
    n_nodes = len(node_names)

    smi_matrix = np.zeros((n_nodes, n_nodes))

    for i, node1 in enumerate(node_names):
        for j, node2 in enumerate(node_names):

            if i == j and hide_diag: continue
            if i!= j and hide_singleclus and len(np.unique(node_idl[node1])) == 1 and  len(np.unique(node_idl[node2])) == 1: continue

            if soft:  smi_matrix[i, j] = soft_mutual_information(node_pprobas[node1],  node_pprobas[node2])
            else:  smi_matrix[i, j] = adjusted_mutual_info_score(node_idl[node1], node_idl[node2])
            if hide_insig:
                if soft and  smi_matrix[i, j] < expected_smi(node_pprobas[node1],  node_pprobas[node2] , len(node_idl[node1]))[0]: smi_matrix[i, j] =0
                if not soft and smi_matrix[i, j] < exp_mutual_info_score(node_idl[node1], node_idl[node2])[0]: smi_matrix[i, j] =0


    # Plot heatmap for the soft mutual information matrix
    plt.figure(figsize=figsz)

    import seaborn as sns
    sns.heatmap(smi_matrix, annot=annot, cmap=cmap, xticklabels=node_names, yticklabels=node_names)
    plt.title("SMI pairw") if soft else plt.title("AMI pairw")
    plt.tight_layout()

    # Save or show the plot
    if svfl is not None:
        os.makedirs(svfl, exist_ok=True)
        plt.savefig(svfl + f"visu_pair_smi_{method_idf}.png")
        plt.close()
    else:
        plt.show()



def plot_asmi_vs_cluster_std(std_values, n_samples=300, trials=30, random_state=0):
    from sklearn.datasets import make_blobs

    asmis = []
    for std in std_values:
        centers = [[0, 0], [1, 1], [5, 5]]  # fixed setup: one close pair, one distant
        X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=std, random_state=random_state)
        gmm1 = GaussianMixture(n_components=3, random_state=random_state).fit(X)
        gmm2 = GaussianMixture(n_components=3, random_state=random_state + 1).fit(X)
        pa = gmm1.predict_proba(X)
        pb = gmm2.predict_proba(X)
        asmi = soft_adjusted_mutual_information(pa, pb, trials=trials)
        asmis.append(asmi)

    plt.figure(figsize=(6, 4))
    plt.plot(std_values, asmis, marker='o')
    plt.title("ASMI vs. Cluster Standard Deviation")
    plt.xlabel("Cluster Std (Overlap ↑)")
    plt.ylabel("Adjusted Soft MI")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def plot_expected_smi_heatmap(k=3, N=200, trials=30):
    confs = np.linspace(0.1, 1.0, 20)
    heatmap = np.zeros((len(confs), len(confs)))

    for i, ca in enumerate(confs):
        for j, cb in enumerate(confs):
            smi, _ = expected_smi(k, k, ca, cb, N=N, trials=trials)
            heatmap[j, i] = smi  # [row, col] = [cb, ca]

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, extent=[0.1, 1.0, 0.1, 1.0], origin='lower', cmap='magma', aspect='auto')
    plt.colorbar(label="Expected Soft MI")
    plt.xlabel("Confidence in Clustering A")
    plt.ylabel("Confidence in Clustering B")
    plt.title("Expected SMI vs Confidence Levels")
    plt.tight_layout()
    plt.show()

