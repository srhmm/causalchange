import os

import matplotlib.pyplot as plt
import numpy as np


def visu_clustering_for_node(node_i, pa_i, X, true_Z_i, node_k, svfl):
    """
    Visualize true vs estimated clusterings for a node and its parents.
    """
    os.makedirs(svfl, exist_ok=True)
    fig, axes = plt.subplots(len(pa_i) + 2, 2, figsize=(14, 10))  # Two columns for each (true, estimated)

    # True clustering for node
    true_labels = true_Z_i
    axes[0, 0].set_title(f"True Clustering: {node_i}")
    # axes[0, 0].hist(X[:, node_i], bins=30, density=True, alpha=0.5, label="True", color='blue')

    cmap_i = plt.cm.get_cmap('tab10', len(np.unique(true_labels)))
    axes[0, 0].scatter(
        np.random.normal(size=X[:, node_i].shape), #np.linspace(min(X[:, node_i]), max(X[:, node_i]), len(X[:, node_i])),
        X[:, node_i], c=true_labels,
        cmap=cmap_i
    )

    def create_legend(ax, labels, cmap):
        """Create a categorical legend for a plot."""
        unique_labels = np.unique(labels)
        handles = [
            plt.Line2D([0], [0], marker='o', color=cmap(label), linestyle='', label=f'Class {label}')
            for label in unique_labels
        ]
        ax.legend(handles=handles, title="Classes", loc="upper right", fontsize='small')

    # i_ax.set_title(f"i ({conf_status_i})")
    create_legend(axes[0, 0], true_labels, cmap_i)

    # Estimated clustering for node
    estimated_labels = node_k
    axes[0, 1].set_title(f"Estimated Clustering: {node_i}")
    # axes[0, 1].hist(X[:, node_i], bins=30, density=True, alpha=0.5, label="Estimated", color='green')

    cmap_i = plt.cm.get_cmap('tab10', len(np.unique(estimated_labels)))
    axes[0, 1].scatter(
        np.linspace(min(X[:, node_i]), max(X[:, node_i]), len(X[:, node_i])), X[:, node_i], c=true_labels,
        cmap=cmap_i
    )
    create_legend(axes[0, 1], estimated_labels, cmap_i)
    # KDE plot for true clustering
    # kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X[:, node_i].reshape(-1, 1))
    # true_density = np.exp(kde.score_samples(np.linspace(min(X[:, node_i]), max(X[:, node_i]), 1000).reshape(-1, 1)))

    unique_labels = np.unique(true_labels)
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))
    import seaborn as sns
    for label in unique_labels:
        sns.kdeplot(
            X[true_labels == label, node_i],
            ax=axes[1, 0],
            fill=True,
            label=f"Class {label}",
            alpha=0.6,
            linewidth=1.5
        )

    # kde_ax.set_title(f"Node {node_idx} ({change_status})", fontsize=10)
    #    kde_ax.tick_params(axis='both', which='major', labelsize=8)
    #    kde_ax.legend(title="Classes", fontsize='x-small')

    # axes[1, 0].plot(np.linspace(min(X[:, node_i]), max(X[:, node_i]), 1000), true_density, label="True KDE",
    #                color='blue')

    # KDE plot for estimated clustering
    # estimated_density = np.exp(
    #    kde.score_samples(np.linspace(min(X[:, node_i]), max(X[:, node_i]), 1000).reshape(-1, 1)))
    # axes[1, 1].plot(np.linspace(min(X[:, node_i]), max(X[:, node_i]), 1000), estimated_density,
    #                 label="Estimated KDE", color='green')
    unique_labels = np.unique(estimated_labels)
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))
    import seaborn as sns
    for label in unique_labels:
        sns.kdeplot(
            X[estimated_labels == label, node_i],
            ax=axes[1, 1],
            fill=True,
            label=f"Class {label}",
            alpha=0.6,
            linewidth=1.5
        )

    # Parent plots (true vs estimated for each parent)
    for idx, parent in enumerate(pa_i):
        # Left column: true clustering for each parent
        axes[2 + idx, 0].scatter(X[:, parent], X[:, node_i], c=true_labels, cmap='viridis')
        axes[2 + idx, 0].set_title(f"Parent {parent} (True)")

        axes[2 + idx, 1].scatter(X[:, parent], X[:, node_i], c=estimated_labels, cmap='viridis')
        axes[2 + idx, 1].set_title(f"Parent {parent}  ")

    plt.tight_layout()
    plt.savefig(svfl + f"node_{node_i}")
    plt.close()
def visu_G_pairwise(X, t_A, t_conf_ix, t_Z, svdir, filename="graph.png"):
    """
    Visualizes the graph and adds a column below it for each node with its KDE and parent-child scatterplots.

    Parameters:
    - X: Dataset (numpy array).
    - t_A: Adjacency matrix representing causal relationships.
    - t_conf_ix: List of sets where each set represents a group of confounded nodes.
    - t_Z: List of label lists for confounded groups.
    - svdir: Directory where the figure should be saved.
    - filename: Name of the output image file.
    """
    import networkx as nx
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    from networkx.drawing.nx_agraph import graphviz_layout

    # Helper function to get labels for a node
    def get_labels(node):
        for k, nset in enumerate(t_conf_ix):
            if node in nset:
                return t_Z[k]
        return np.zeros(X.shape[0])

    # Create the graph
    G = nx.DiGraph()

    n_nodes = X.shape[1]  # Number of regular nodes
    n_conf_sets = len(t_Z)  # Number of confounding groups

    # Add nodes
    for i in range(n_nodes):
        G.add_node(i, type='regular')
    for k in range(n_conf_sets):
        G.add_node(f"Z{k + 1}", type='confounder')

    # Add edges from adjacency matrix
    for i in range(n_nodes):
        for j in range(n_nodes):
            if t_A[i][j] != 0:
                G.add_edge(i, j)

    # Add confounding edges
    for k, nset in enumerate(t_conf_ix):
        for i in nset:
            G.add_edge(f"Z{k + 1}", i)

    # Initialize the figure with a proportionally larger height
    fig = plt.figure(figsize=(14, 18))  # Increase height for better plot visibility

    # Adjust the GridSpec for more rows with optimized height ratios
    n_rows = 6  # Total rows in the grid
    gs = GridSpec(
        n_rows, n_nodes,
        height_ratios=[1.5] + [1] * (n_rows - 1),  # First row (graph) gets more height
        figure=fig,
        hspace=0.8,  # Reduce vertical space between rows
        wspace=0.3  # Reduce horizontal space between columns
    )
    # Graph visualization (Top row)
    graph_ax = fig.add_subplot(gs[0, :])

    # Choose layout
    try:
        pos = graphviz_layout(G, prog="dot")  # Hierarchical layout
    except ImportError:
        pos = nx.spring_layout(G, seed=42, k=1.5)  # Spring layout as fallback

    # Assign colors based on type
    node_colors = [
        'lightblue' if G.nodes[node]['type'] == 'regular' else 'lightcoral' for node in G.nodes
    ]

    nx.draw(
        G, pos, ax=graph_ax, with_labels=True, node_color=node_colors, edge_color="gray",
        node_size=800, font_size=10, arrows=True
    )
    graph_ax.set_title("Graph Visu")

    # Columns for each node (Bottom row)
    for node_idx in range(n_nodes):
        # Determine confounding/changing status
        change_status = "changing" if any(node_idx in nset for nset in t_conf_ix) else "--"

        # Get parents of the node
        pa_i = [k for k in np.where(t_A[:, node_idx] != 0)[0]]
        pa_i_labels = [get_labels(k) for k in pa_i]

        # First row: KDE plot for the node
        kde_ax = fig.add_subplot(gs[1, node_idx])
        labels = get_labels(node_idx)
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))

        for label in unique_labels:
            sns.kdeplot(
                X[labels == label, node_idx],
                ax=kde_ax,
                fill=True,
                label=f"Class {label}",
                alpha=0.6,
                linewidth=1.5
            )

        kde_ax.set_title(f"Node {node_idx} ({change_status})", fontsize=10)
        kde_ax.tick_params(axis='both', which='major', labelsize=8)
        kde_ax.legend(title="Classes", fontsize='x-small')

        # Add parent plots below the KDE plot
        for parent_idx, parent in enumerate(pa_i):
            if parent_idx + 2 >= gs.nrows:
                continue  # Prevent exceeding grid space
            parent_ax = fig.add_subplot(gs[parent_idx + 2, node_idx])
            labels_k = pa_i_labels[parent_idx]
            cmap_k = plt.cm.get_cmap('tab10', len(np.unique(labels_k)))
            scatter = parent_ax.scatter(X[:, parent], X[:, node_idx], c=labels_k, cmap=cmap_k)

            conf_status_k = "jointly cfd" if any(
                parent in nset and node_idx in nset for nset in t_conf_ix
            ) else "changing pa" if any(parent in nset for nset in t_conf_ix) else "--"

            parent_ax.set_title(f"Parent {parent} ({conf_status_k})", fontsize=8)
            parent_ax.tick_params(axis='both', which='major', labelsize=7)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(svdir + "/" + filename)
    plt.close()


def visu_G_with_marginals(X, t_A, t_conf_ix, t_Z, svdir, filename="graph_visualization_kde.png"):
    """
    Visualizes the entire graph and KDE plots for each node.

    Parameters:
    - X: Dataset (numpy array).
    - t_A: Adjacency matrix representing causal relationships.
    - t_conf_ix: List of sets where each set represents a group of confounded nodes.
    - t_Z: List of label lists for confounded groups.
    - svdir: Directory where the figure should be saved.
    - filename: Name of the output image file.
    """
    import networkx as nx
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    from networkx.drawing.nx_agraph import graphviz_layout

    # Helper function to get labels for a node
    def get_labels(node):
        for k, nset in enumerate(t_conf_ix):
            if node in nset:
                return t_Z[k]
        return np.zeros(X.shape[0])

    # Create the graph
    G = nx.DiGraph()

    n_nodes = X.shape[1]  # Number of regular nodes
    n_conf_sets = len(t_Z)  # Number of confounding groups

    # Add nodes
    for i in range(n_nodes):
        G.add_node(i, type='regular')
    for k in range(n_conf_sets):
        G.add_node(f"Z{k + 1}", type='confounder')

    # Add edges from adjacency matrix
    for i in range(n_nodes):
        for j in range(n_nodes):
            if t_A[i][j] != 0:
                G.add_edge(i, j)

    # Add confounding edges
    for k, nset in enumerate(t_conf_ix):
        for i in nset:
            G.add_edge(f"Z{k + 1}", i)

    # Initialize the figure
    fig = plt.figure(figsize=(14, 12))  # Adjusted height for more space
    gs = GridSpec(2, 1, height_ratios=[1, 3], figure=fig)  # Separate rows for graph and plots

    # Graph visualization (Top row)
    graph_ax = fig.add_subplot(gs[0, 0])

    # Choose layout
    try:
        pos = graphviz_layout(G, prog="dot")  # Hierarchical layout
    except ImportError:
        pos = nx.spring_layout(G, seed=42, k=1.5)  # Spring layout as fallback

    # Assign colors based on type
    node_colors = [
        'lightblue' if G.nodes[node]['type'] == 'regular' else 'lightcoral' for node in G.nodes
    ]

    nx.draw(
        G, pos, ax=graph_ax, with_labels=True, node_color=node_colors, edge_color="gray",
        node_size=800, font_size=10, arrows=True
    )
    graph_ax.set_title("Graph with Nodes and Confounding Edges")

    # KDE plots for each node (Bottom row)
    grid_cols = 4
    grid_rows = (n_nodes + grid_cols - 1) // grid_cols  # Ensure enough rows for all nodes
    gs2 = GridSpec(grid_rows, grid_cols, figure=fig, top=0.5, bottom=0.05, hspace=0.6, wspace=0.5)

    for idx in range(n_nodes):
        labels = get_labels(idx)
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))
        ax = fig.add_subplot(gs2[idx // grid_cols, idx % grid_cols])

        # Plot KDE with class-based coloring
        for label in unique_labels:
            sns.kdeplot(
                X[labels == label, idx],
                ax=ax,
                fill=True,
                label=f"Class {label}",
                alpha=0.6,
                linewidth=1.5
            )

        ax.set_title(f"Node {idx}", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(title="Classes", fontsize='x-small')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(svdir + "/" + filename)
    plt.close()


def visu_pair(X, t_A, t_conf_ix, i, j, t_Z, tru, deci,  Zi, Zj, Zij, Zji, svdir):
    """
    Creates a figure with multiple subplots, dynamically adjusting for the number of parents of `i`.
    Saves the figure to the specified directory.

    Parameters:
    - X: Dataset (numpy array).
    - t_A: Adjacency matrix representing causal relationships.
    - t_conf_ix: List of sets where each set represents a group of confounded nodes.
    - i, j: Indices of the nodes being analyzed.
    - t_Z: List of label lists for confounded groups.
    - tru: Ground truth relationship between i and j (e.g., 'true', 'false').
    - deci: Decision made for the relationship.
    - svdir: Directory where the figure should be saved.
    """
    from matplotlib.gridspec import GridSpec

    def get_labels(node):
        """Retrieve clustering labels for a given node."""
        for k, nset in enumerate(t_conf_ix):
            if node in nset:
                return t_Z[k]
        return np.zeros(X.shape[0])

    def create_legend(ax, labels, cmap):
        """Create a categorical legend for a plot."""
        unique_labels = np.unique(labels)
        handles = [
            plt.Line2D([0], [0], marker='o', color=cmap(label), linestyle='', label=f'Class {label}')
            for label in unique_labels
        ]
        ax.legend(handles=handles, title="Classes", loc="upper right", fontsize='small')

    # Confounding status for i and j
    conf_status_i = "jointly cfd" if any(i in nset and j in nset for nset in t_conf_ix) else "changing" if any(
        i in nset for nset in t_conf_ix) else "--"
    conf_status_j = "jointly cfd" if any(j in nset and i in nset for nset in t_conf_ix) else "changing" if any(
        j in nset for nset in t_conf_ix) else "--"

    # Labels for i and j
    labels_i = get_labels(i)
    labels_j = get_labels(j)

    # Parents of i and j
    pa_i = [k for k in np.where(t_A[:, i] != 0)[0] if k != j]
    pa_i_labels = [get_labels(k) for k in pa_i]

    pa_j = [k for k in np.where(t_A[:, j] != 0)[0] if k != i]
    pa_j_labels = [get_labels(k) for k in pa_j]

    n_rows = 4 + max(len(pa_i), len(pa_j))  # 1 row for `i`, rows for its parents, and 1 for `i->j`

    fig = plt.figure(figsize=(8, 4 + 2 * n_rows))
    gs = GridSpec(n_rows, 2, figure=fig, width_ratios=[2, 2])

    # Plot `i`
    i_ax = fig.add_subplot(gs[0, 0])
    cmap_i = plt.cm.get_cmap('tab10', len(np.unique(labels_i)))
    i_ax.scatter(
        np.random.normal(size=X[:, i].shape), #np.linspace(min(X[:, i]), max(X[:, i]), len(X[:, i])),
        X[:, i], c=labels_i, cmap=cmap_i
    )
    i_ax.set_title(f"i ({conf_status_i}) -- true labels")
    create_legend(i_ax, labels_i, cmap_i)

    i_ax = fig.add_subplot(gs[1, 0])
    cmap_i = plt.cm.get_cmap('tab10', len(np.unique(Zi)))
    i_ax.scatter(
        np.random.normal(size=X[:, i].shape), #np.linspace(min(X[:, i]), max(X[:, i]), len(X[:, i])),
        X[:, i], c=Zi, cmap=cmap_i
    )
    i_ax.set_title(f"i ({conf_status_i}) -- estimated")
    create_legend(i_ax, labels_i, cmap_i)

    # Plot `i->j`
    # ij_ax = fig.add_subplot(gs[len(pa_i) + 1, 0])
    ij_ax = fig.add_subplot(gs[2, 0])
    title_ij = f'i -> j '
    if t_A[i][j] != 0:
        title_ij += " (TRUE)"
    scatter = ij_ax.scatter(X[:, i], X[:, j], c=labels_i, cmap=cmap_i)
    ij_ax.set_title(title_ij)
    create_legend(ij_ax, labels_i, cmap_i)

    ij_ax = fig.add_subplot(gs[3, 0])
    title_ij = title_ij + "-- estimated"
    ij_ax.scatter(X[:, i], X[:, j], c=Zij, cmap=cmap_i)
    ij_ax.set_title(title_ij)
    create_legend(ij_ax, labels_i, cmap_i)

    # Plot parents of `i`
    for idx, parent in enumerate(pa_i):
        parent_ax = fig.add_subplot(gs[idx + 4, 0])
        labels_k = pa_i_labels[idx]
        cmap_k = plt.cm.get_cmap('tab10', len(np.unique(labels_k)))
        scatter = parent_ax.scatter(X[:, parent], X[:, i], c=labels_k, cmap=cmap_k)
        conf_status_k = "jointly cfd" if any(
            parent in nset and i in nset for nset in t_conf_ix) else "changing pa" if any(
            parent in nset for nset in t_conf_ix) else "--"
        parent_ax.set_title(f"i's parent {parent} ({conf_status_k})")
        create_legend(parent_ax, labels_k, cmap_k)

    # Plot `j`
    j_ax = fig.add_subplot(gs[0, 1])
    cmap_j = plt.cm.get_cmap('tab10', len(np.unique(labels_j)))
    scatter = j_ax.scatter(
        np.random.normal(size=X[:, j].shape),#np.linspace(min(X[:, j]), max(X[:, j]), len(X[:, j])),
        X[:, j], c=labels_j, cmap=cmap_j
    )
    j_ax.set_title(f"j")
    create_legend(j_ax, labels_j, cmap_j)

    j_ax = fig.add_subplot(gs[1, 1])
    cmap_j = plt.cm.get_cmap('tab10', len(np.unique(Zj)))
    scatter = j_ax.scatter(
        np.random.normal(size=X[:, j].shape),#np.linspace(min(X[:, j]), max(X[:, j]), len(X[:, j])),
        X[:, j], c=Zj, cmap=cmap_j
    )
    j_ax.set_title(f"j --estimated")
    create_legend(j_ax, labels_j, cmap_j)

    # Plot `j->i`
    ji_ax = fig.add_subplot(gs[2, 1])
    title_ji = f'j -> i ({conf_status_j})'
    if t_A[j][i] != 0:
        title_ji += " (TRUE)"
    scatter = ji_ax.scatter(X[:, j], X[:, i], c=labels_j, cmap=cmap_j)
    ji_ax.set_title(title_ji)
    create_legend(ji_ax, labels_j, cmap_j)
    # Plot `j->i`
    ji_ax = fig.add_subplot(gs[3, 1])
    title_ji = f'j -> i ({conf_status_j}) -- estimated'
    scatter = ji_ax.scatter(X[:, j], X[:, i], c=Zji, cmap=cmap_j)
    ji_ax.set_title(title_ji)
    create_legend(ji_ax, labels_j, cmap_j)

    # Plot parents of `j`
    for idx, parent in enumerate(pa_j):
        parent_ax = fig.add_subplot(gs[idx + 4, 1])
        labels_k = pa_j_labels[idx]
        cmap_k = plt.cm.get_cmap('tab10', len(np.unique(labels_k)))
        scatter = parent_ax.scatter(X[:, parent], X[:, j], c=labels_k, cmap=cmap_k)
        conf_status_k = "jointly cfd" if any(
            parent in nset and j in nset for nset in t_conf_ix) else "changing pa" if any(
            parent in nset for nset in t_conf_ix) else "--"
        parent_ax.set_title(f"j's parent {parent} ({conf_status_k})")
        create_legend(parent_ax, labels_k, cmap_k)

    # Add column titles
    fig.text(0.2, 0.95, f'i: {conf_status_i}', fontsize=14, ha='center', va='top')
    fig.text(0.7, 0.95, f'j: {conf_status_j}', fontsize=14, ha='center', va='top')

    # Add supertitle
    plt.suptitle(f"True: {tru} Deci: {deci}", fontsize=16, y=0.98)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(svdir + f"E_{i}_{j}_{tru}.png")
    plt.close()
