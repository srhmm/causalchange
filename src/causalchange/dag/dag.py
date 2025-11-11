import numpy as np
from collections import defaultdict
from typing import List

from src.causalchange.util.upq import UPQ
from src.causalchange.util.utils import is_insignificant

#from src.causalchange import score_edge_continuous
#from src.causalchange import ContinuousScoringFunction
#from src.causalchange.utils.util import is_insignificant



class DAG:
    def __init__(self, data, n_n, edges_state, **optargs):
        r""" DAG model.

        :param data: datasets, `data[i]` of shape `DxN` with `D` entries, `N` nodes, in dataset `i`.
        :param optargs: optional arguments

        :Keyword Arguments:
        * *scoring_function* (``ScoringFunction``) -- scoring function, regressor type
        * *is_true_edge* (``(i: int,int) -> (j: int) -> str``) -- ground truths for logging
        * *lg* (``logging``) -- logger if verbosity>0
        * *verbosity* (``int``) -- verbosity level
        """
        #self.n_n = data[0].shape[1]
        #self.n_c = len(data)
        self.n_n = n_n
        self.data_C = data
        self.nodes = set([i for i in range(self.n_n)])

        self.edges_state = edges_state

        # Optional hyperparameters
        #self.scoring_function = optargs.get("scoring_function", ContinuousScoringFunction.GP)
        self.is_true_edge = optargs.get("is_true_edge", lambda i: lambda j: "")
        self.lg = optargs.get("lg", None)
        self.verbosity = optargs.get("verbosity", 0)

        # each node Y with its current parents {X1,...Xn}
        #self.node_mdl = defaultdict(float)
        self.node_pa = defaultdict(set)
        self.node_ch = defaultdict(set)
        self.pair_edges = [[None for _ in range(self.n_n)] for _ in range(self.n_n)]
        #self.mdl_cache = {}
        #self.cps_cache = {}

    def children_of(self, j):
        return [i for i in self.nodes if self.is_edge(j, i)]

    def parents_of(self, j):
        return [i for i in self.nodes if self.is_edge(i, j)]

    def get_nodes(self):
        return self.nodes

    def get_adj(self):
        return np.array([
            [int(self.is_edge(i, j)) for j in self.nodes]
            for i in self.nodes
        ])

    def is_edge(self, i, j):
        return i in self.node_pa[j]

    def _score_graph(self):
        return self.eval_other_dag(self.get_adj())

    def add_edge(self, i, j: int, score: int, gain: int, silence=False):
        """
        DAG: Add edge Xi -> Xj.

        :param i: parent, (Xi, lag) or i:int
        :param j: target, convention lag 0
        :param score: score(parents(Xj) + {Xi}, Xj).
        :param gain: score(parents(Xj), Xj)-score(parents(Xj) + {Xi}, Xj).
        :return:
        """
        #self.node_mdl[j] = score
        self.node_pa[j].add(i)
        self.node_ch[i].add(j)

        if self.verbosity > 0 and not silence:
            self.lg.info(
                f'\tAdding edge {i} -> {j}: s={np.round(gain, 2)}'  
                f'\t{self.is_true_edge(i)(j)}')

    def remove_edge(self, i, j: int, silence=False):
        """
        Remove Xi -> Xj. This will also update the score for Xj to be that of Xpa(j) setminus Xi -> Xj.

        :param i: parent
        :param j: child
        :return:
        """
        assert (i in self.node_pa[j])
        self.node_pa[j].remove(i)
        self.node_ch[i].remove(j)
        pa_up = self.parents_of(j)
        #self.node_mdl[j] = self.eval_edge(j, pa_up)

        if self.verbosity > 0 and not silence:
            self.lg.info(
                f'\tRemoving edge {i} -> {j}\t{self.is_true_edge(i)(j)}')

    def remove_all_edges(self):
        for i in self.nodes:
            for j in self.nodes:
                if not (i in self.node_pa[j]):
                    continue
                self.remove_edge(i, j)

    """ Scoring """

    def eval_edge(self, j: int, pa: List) -> int:
        """
        Evaluates MDL score for a causal relationship pa(Xj)->Xj.

        :param j: Xj, node index
        :param pa: pa(Xj), list of node indices
        :return: score(pa(Xj)->Xj)
        """
        score, res_dict = self.edges_state.score_edge(j, pa)
        return score


    """ Search - score pairwise edges initially """

    def initial_edges(self, q: UPQ, skip_insignificant=False) -> UPQ:
        for j in self.nodes:
            pa = []
            score = self.eval_edge(j, pa)
            others = [i for i in self.nodes if not (i == j)]
            for i in others:
                score_ij = self.eval_edge(j, [i])

                edge_ij = QEntry(i, j, score_ij, score)

                gain = score - score_ij
                prio = gain * 100
                if (not skip_insignificant) or (not is_insignificant(gain)):
                    q.add_task(task=edge_ij, priority=prio)
                self.pair_edges[i][j] = edge_ij
        return q

    """ Search - evaluate local modifications """

    def eval_edge_addition(self, j, i, return_score=False):
        """ DAG: Gain of adding edge Xi->Xj.

        :param j: target
        :param i: parent
        :return:
        """
        pa_cur = self.parents_of(j)
        pa_up = pa_cur.copy()
        pa_up.append(i)

        score_cur = self.eval_edge(j, pa_cur)
        score_up = self.eval_edge(j, pa_up)

        gain = (score_cur - score_up)
        if return_score:
            return gain, score_up
        else:
            return gain

    def eval_edge_flip(self, j, ch):
        """ current edge j ->ch, Evaluates {j} <- {ch},pa_j against {j} u pa_ch -> pa_j """
        pa_j_cur = self.parents_of(j)
        pa_ch_cur = self.parents_of(ch)
        assert j in pa_ch_cur

        pa_j_up = pa_j_cur.copy()
        pa_j_up.append(ch)

        pa_ch_up = pa_ch_cur.copy()
        pa_ch_up.remove(j)

        score_cur = self.eval_edge(j, pa_j_cur) + self.eval_edge(ch, pa_ch_cur)
        score_up = self.eval_edge(j, pa_j_up) + self.eval_edge(ch, pa_ch_up)

        gain = (score_cur - score_up)

        if self.verbosity > 0:
            self.lg.info(
                f'\tEval edge flip {j} -> {ch}: s={np.round(score_cur, 2)} to {j} <- {ch}:'
                f' s={np.round(score_up, 2)}  \t {self.is_true_edge(ch)(j)}')

        return gain

    def eval_edges(self, j, new_parents):
        """
        Gain of replacing causal parents of Xj

        :param j: target
        :param new_parents: new parent set
        :return:
        """
        old_score = self.eval_edge(j, self.parents_of(j))
        new_score = self.eval_edge(j, new_parents)
        gain = old_score - new_score
        return gain, new_score

    def update_edges(self, j, new_parents):
        """ Update graph around Xj

        :param j: target Xj
        :param new_parents: new parent set
        :return:
        """
        old_parents = self.parents_of(j)
        for i in old_parents:
            self.remove_edge(i, j, silence=i in new_parents)  # no need to print the removal if i remains a parent

        for i in new_parents:
            gain, score = self.eval_edge_addition(j, i, return_score=True)
            self.add_edge(i, j, score, gain, silence=True)

    def eval_other_dag(self, adj):
        """ Evaluate the MDL score for a given DAG *(regardless the edges in this DAG object).*"""
        mdl = 0
        for j in self.nodes:
            pa = [i for i in self.nodes if adj[i][j] != 0]
            score_j = self.eval_edge(j, pa)
            mdl = mdl + score_j
        return mdl

    def has_cycle(self, i, j):  # from https://www.geeksforgeeks.org/detect-cycle-in-a-graph/
        visited = [False] * (self.n_n + 1)
        recStack = [False] * (self.n_n + 1)
        for node in range(self.n_n):
            if visited[node]:
                if self._has_cycle_util(node, visited, recStack, i, j):
                    return True
        return False

    def _has_cycle_util(self, v, visited, recStack, i, j):
        visited[v] = True
        recStack[v] = True
        neighbors = []
        if v in self.node_ch:
            neighbors = self.node_ch[v]
        if v == i:
            neighbors = [n for n in range(self.n_n) if n in neighbors or n == j]

        for neighbour in neighbors:
            if not visited[neighbour]:
                if self._has_cycle_util(neighbour, visited, recStack, i, j):
                    return True
            elif recStack[neighbour]:
                return True

        recStack[v] = False
        return False

    def exists_anticausal_edge(self, parent, node):
        if self.is_edge( node , parent ):
            return True
        return False

class QEntry:
    def __init__(self, i, j, score_ij, score_0):
        self.i = i
        self.pa = i
        self.j = j

        # Score of edge i->j in the empty graph
        self.score_ij = score_ij
        # Score of []->j
        self.score_0 = score_0

    def __hash__(self):
        return hash((self.i, self.j))

    def __eq__(self, other):
        return (self.i == other.i
                & self.j == other.j)

    def __str__(self):
        return f'j_{str(self.i)}_i_{str(self.j)}'
