from __future__ import annotations

from typing import Any

from causalchange.core.require import require_matplotlib


def _temporal_node_label(node: Any) -> str:
    if isinstance(node, tuple) and len(node) == 2:
        var, lag = node
        return f"{var}(t)" if int(lag) == 0 else f"{var}(t-{lag})"

    return str(node)


def plot_graph(graph, *, title: str | None = None, ax=None, seed: int = 42):
    import networkx as nx

    plt = require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    pos = nx.spring_layout(graph, seed=seed)

    nx.draw(
        graph,
        pos=pos,
        ax=ax,
        with_labels=True,
        node_size=1400,
        font_size=9,
        arrows=True,
    )

    if title:
        ax.set_title(title)

    return ax


def plot_temporal_graph(graph, *, title: str | None = None, ax=None, seed: int = 42):
    import networkx as nx

    plt = require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    pos = nx.spring_layout(graph, seed=seed)
    labels = {node: _temporal_node_label(node) for node in graph.nodes}

    nx.draw(
        graph,
        pos=pos,
        labels=labels,
        ax=ax,
        with_labels=True,
        node_size=1500,
        font_size=9,
        arrows=True,
    )

    if title:
        ax.set_title(title)

    return ax
