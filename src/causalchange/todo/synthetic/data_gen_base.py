import os
import random

import scipy as sp
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


class CausalDataGen(object):
    def __init__(self, N, M, **kwargs):
        """
        Generates data from an SCM.

        :param N: n samples
        :param M: n nodes
        :param kwargs: optional parameters: gen in {'lin', 'quad', 'cub', 'exp', 'log', 'sin', 'mix'}, seed: int, verbosity: int
        :return:
        """
        self.N = N
        self.M = M
        self.seed = kwargs.get('seed', 42)
        self.gen_dict = \
            {'lin': lambda x: x,
             'quad': lambda x: x ** 2,
             'cub': lambda x: x ** 3,
              'exp': np.exp, #some overflow problems?
             # 'log': sp.special.expit,
             'sin': np.sin,
             'mix': np.random.choice(
                 [lambda x: x, lambda x: x ** 2, lambda x: x ** 3, lambda x: x ** 4, np.sin], 1)[0]
             }

        self.gen_type = kwargs.get('gen', 'lin')
        self.noise_type = kwargs.get('noise', 'normal')
        self.dag_type = kwargs.get('dag', 'scale_free')
        self.vb = kwargs.get('verbosity', 0)

    def gen_X(self):
        self._gen_dag()
        X = self._gen_noise()
        X = self._gen_functional_deps(X)
        return X

    def _gen_dag(self):
        # Adjacency
        def _graph():
            if self.dag_type == 'scale_free':
                return self._gen_scale_free()
            else:
                raise ValueError(self.dag_type)

        self.G = _graph()
        A = self._graph_to_adj()

        # Causal weights, make sure bounded away from 0
        intervals = [[-0.75, -0.25], [0.25, 0.75]]
        mask = np.asarray([[random.uniform(
            *random.choices(intervals, weights=[r[1] - r[0] for r in intervals])[0]) for _ in range(self.M)] for _ in
            range(self.M)])
        # mask = np.random.uniform(0.25, 0.75 [self.M, self.M])
        W = A * mask
        self._A = A
        # Weights of additive components
        self.W = W
        self.w = np.random.uniform(0, 1, self.M)
        # Bias
        self.u = np.random.uniform(0, 1, self.M)

    def _graph_to_adj(self):
        A = np.zeros([self.M, self.M])
        for u, v in self.G.edges:
            A[u, v] = 1
        return A

    def _gen_functional_deps(self, X):
        fun = self.gen_dict[self.gen_type]
        G_sort = nx.topological_sort(self.G)
        for i in G_sort:
            X = self._gen_functional_dep_i(X, i, fun)
        return X

    def _gen_functional_dep_i(self, X, i, fun):
        par = list(self.G.predecessors(i))
        X[:, i] += fun(  # self.w[i] *
            X[:, par].dot(self.W[par, i])) + self.u[i]
        return X

    def _gen_noise(self):
        X = np.zeros((self.N, self.M))

        def _noise(loc, sc, sz):
            if self.noise_type == 'normal':
                Xi = np.random.normal(loc, sc, sz)
            elif self.noise_type == 'exp':
                Xi = np.random.exponential(sc, sz) + loc
            elif self.noise_type == 'gumbel':
                Xi = np.random.gumbel(loc, sc, sz)
            else:
                Xi = np.zeros(self.N)
            return Xi

        for i in self.G:
            X[:, i] = _noise(0, 1, self.N)

        return X

    def _gen_scale_free(self):
        G = nx.directed.scale_free_graph(
            self.M,
            alpha=0.41,
            beta=0.54,
            gamma=0.05,
            delta_in=0.2,
            delta_out=0)
        G = G.to_directed()
        _G = nx.DiGraph()
        for u, v, _ in G.edges:
            if (u, v) not in _G.edges:
                _G.add_edge(u, v)
        G = _G
        try:
            while True:
                cycle = nx.find_cycle(G)
                e = cycle.pop()
                G.remove_edge(*e)
        except nx.NetworkXNoCycle:
            pass
        return G

    def plot_X(self, X, plot_dir=None):
        for i in self.G:
            pa_i = list(self.G.predecessors(i))
            fig, axes = plt.subplots(len(pa_i) + 1, 1, figsize=(14, 10))

            true_labels = np.zeros(X.shape[0])
            ax1 = axes if len(pa_i) < 1 else axes[0]
            ax1.set_title(f"Node {i}: P(X{i})")

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
            if plot_dir is not None:
                os.makedirs(plot_dir, exist_ok=True)
                plt.savefig(plot_dir + f"node_{i}_pa_{pa_i}")
                plt.close()
            else:
                plt.show()
        if plot_dir is not None:
            print(f"Example plots saved in {plot_dir}")
