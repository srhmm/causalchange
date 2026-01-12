from __future__ import annotations

from dataclasses import dataclass
from typing import Set, Tuple

import networkx as nx


Edge = Tuple[str, str]
Undir = Tuple[str, str]


def directed_edges(G: nx.DiGraph) -> Set[Edge]:
    return {(str(u), str(v)) for (u, v) in G.edges() if u != v}


def skeleton_edges(G: nx.DiGraph) -> Set[Undir]:
    out: Set[Undir] = set()
    for (u, v) in directed_edges(G):
        a, b = (u, v) if u < v else (v, u)
        out.add((a, b))
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


@dataclass(frozen=True)
class GraphMetrics:
    edge_precision: float
    edge_recall: float
    edge_f1: float
    skel_precision: float
    skel_recall: float
    skel_f1: float
    shd: int



def edge_f1(true_g: nx.DiGraph, est_g: nx.DiGraph) -> tuple[float, float, float]:
    t = directed_edges(true_g)
    e = directed_edges(est_g)
    tp = len(t & e)
    fp = len(e - t)
    fn = len(t - e)
    return prf(tp, fp, fn)


def skeleton_f1(true_g: nx.DiGraph, est_g: nx.DiGraph) -> tuple[float, float, float]:
    t = skeleton_edges(true_g)
    e = skeleton_edges(est_g)
    tp = len(t & e)
    fp = len(e - t)
    fn = len(t - e)
    return prf(tp, fp, fn)


def shd(true_g: nx.DiGraph, est_g: nx.DiGraph) -> int:
    t_dir = directed_edges(true_g)
    e_dir = directed_edges(est_g)

    t_skel = skeleton_edges(true_g)
    e_skel = skeleton_edges(est_g)

    add_del = len(t_skel ^ e_skel)

    reversals = 0
    for (a, b) in (t_skel & e_skel):
        t_forward = (a, b) in t_dir
        e_forward = (a, b) in e_dir
        if t_forward != e_forward:
            reversals += 1

    return add_del + reversals


def compute_metrics(true_g: nx.DiGraph, est_g: nx.DiGraph) -> GraphMetrics:
    ep, er, ef1 = edge_f1(true_g, est_g)
    sp, sr, sf1 = skeleton_f1(true_g, est_g)
    return GraphMetrics(
        edge_precision=ep,
        edge_recall=er,
        edge_f1=ef1,
        skel_precision=sp,
        skel_recall=sr,
        skel_f1=sf1,
        shd=shd(true_g, est_g),
    )
