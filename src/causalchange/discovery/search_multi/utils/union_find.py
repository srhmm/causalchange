from __future__ import annotations

from collections.abc import Hashable, Iterable


def union_find_components(
    nodes: list[Hashable],
    edges: Iterable[tuple[Hashable, Hashable]],
) -> list[list[Hashable]]:
    parent = {x: x for x in nodes}
    rank = {x: 0 for x in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in edges:
        union(a, b)

    comps: dict[Hashable, list[Hashable]] = {}
    for x in nodes:
        rx = find(x)
        comps.setdefault(rx, []).append(x)

    return list(comps.values())
