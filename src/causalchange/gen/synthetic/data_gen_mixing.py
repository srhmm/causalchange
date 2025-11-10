import os

import scipy as sp
import networkx as nx
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

from src.causalchange.util.utils_idl import pi_join


def _coin(p, size=1):
    return np.random.binomial(1, p, size=size)

def unif_away_zero(low=.25, high=1, size=1, all_positive=False):
    """ from causaldag package """
    if all_positive:
        return np.random.uniform(low, high, size=size)
    signs = (_coin(.5, size) - .5) * 2
    return signs * np.random.uniform(low, high, size=size)


def unif_away(Ws, is_flip_iv, low=.25, high=1, sep=0.25):
    W_away = np.random.uniform(low, high, size=Ws[0].shape)
    signs = (_coin(.5, Ws[0].shape) - .5) * 2
    W_away *= -np.sign(Ws[len(Ws)-1]) if is_flip_iv  else signs
    if len(W_away.shape)==2:
        for i in range(W_away.shape[0]):
            for j in range(W_away.shape[1]):
                it = 100
                while np.any([abs(W_away[i][j] - W[i][j]) < sep for W in Ws]) and it > 0:
                    it -= 1
                    W_away[i, j] = unif_away_zero()[0]
    else:
        for i in range(W_away.shape[0]):
            it = 100
            while np.any([abs(W_away[i] - W[i]) < sep for W in Ws]) and it > 0:
                it -= 1
                W_away[i] = unif_away_zero()[0]
    return W_away


class DataGen(object):
    # Functional Form
    gen_dict = \
        {'lin': lambda x: x,
         'quad': lambda x: x ** 2,
         'cub': lambda x: x ** 3,
         'exp': np.exp, 'log': sp.special.expit, 'sin': np.sin,
         'mix':
             np.random.choice([lambda x: x, lambda x: x ** 2, lambda x: x ** 3, lambda x: x ** 4, np.exp, np.sin], 1)[
                 0]}

    def __init__(self, params, graph, seed, vb=0):
        self.N = params["S"]
        self.M = params["N"]
        self.gen = params["F"].value
        self.noise = params["NS"].value
        self.dag_gen = params["DG"].value
        self.ivty = params["IVT"].value
        self.NOISE_SC = 0.5

        self.dag_exp_deg = 1
        self.graph = graph
        self._gen_dag_weigths(graph)

        self.b = np.random.uniform(0, 1, self.M)
        self.w = np.random.uniform(0, 1, self.M)
        self.u = np.random.uniform(0, 1, self.M)
        self.e = np.random.randint(1, 5, self.M)
        self.vb = vb

        self.scale_post = False
        self.scale_during = True # iSCM, see  https://arxiv.org/pdf/2406.11601
        assert not self.scale_post and self.scale_during

    def _gen_dag_weigths(self, graph):
        self.G = graph
        A = nx.to_numpy_array(graph)

        # Causal weights, make sure bounded away from 0
        intervals = [[-0.75, -0.25], [0.25, 0.75]]
        mask = np.asarray([[random.uniform(
            *random.choices(intervals, weights=[r[1] - r[0] for r in intervals])[0]) for _ in range(self.M)] for _
            in range(self.M)])
        W = A * mask
        self._A = A
        # Weights of additive components
        self.W = W

    def gen_X(self, t_n_Z, t_Zs):
        self.Zs = t_Zs
        self.conf_ind_sets = t_n_Z
        self._gen_Z_weights()
        X = self._gen_noise()
        X = self._gen_functional_deps(X)
        return X

    def gen_unconfounded_X(self ):
        self.Zs = []
        self.conf_ind_sets = []
        self._gen_Z_weights()
        X = self._gen_noise()
        X = self._gen_functional_deps(X)
        return X

    def _gen_functional_deps(self, X):
        fun = self.gen_dict[self.gen]
        print(fun)
        G_sort = nx.topological_sort(self.G)
        for i in G_sort:
           X = self._gen_functional_dep_i(X, i, fun)
           if self.scale_during: X = preprocessing.StandardScaler().fit(X).transform(X)

        if self.scale_post: X = preprocessing.StandardScaler().fit(X).transform(X)
        return X

    def _gen_functional_dep_i(self, X, i, fun):
        par = list(self.G.predecessors(i))
        if len(par) == 0:
            return X

        is_cf = any([i in conf_ind for conf_ind in self.conf_ind_sets])
        if not is_cf:
            X[:, i] += fun(self.w[i] * X[:, par].dot(self.W[par, i])) + self.u[i]
            return X

        assigned = False
        for iz, conf_ind in enumerate(self.conf_ind_sets):
            if not i in conf_ind:
                continue
            for k in np.unique(self.Zs[iz]):
                X[:, i][self.Zs[iz] == k] += fun((X[:, par][self.Zs[iz] == k]).dot(
                        self.Ws[iz][k][par, i])) + self.us[iz][k][i]
                if self.vb > 0: print( f"Mixing node {i} | {par},\tclass coefs\tW={self.Ws[iz][k][par, i]}")
            assigned = True
        assert assigned
        return X

    def _gen_Z_weights(self):
        self.Ws = {}
        self.ws = {}
        self.bs = {}
        self.us = {}

        for iz, (conf_ind, Z) in enumerate(zip(self.conf_ind_sets, self.Zs)):
            self.Ws[iz] = {0: self.W}
            self.ws[iz] = {0: self.w}
            self.bs[iz] = {0: self.b}
            self.us[iz] = {0: self.u}
            for k in np.unique(Z):
                if k == 0: continue
                # Causal weights, make sure bounded away from 0 and from other clusters
                W_previous =\
                    [self.Ws[jz][m]
                     for jz in self.Ws.keys()
                     for m in self.Ws[jz].keys() if int(jz) < int(iz) or int(m) < int(k)]
                mask = unif_away(W_previous, is_flip_iv=(self.ivty=='flip')) if len(W_previous) else unif_away_zero(size=(self.M, self.M))
                sgn = (-1 if k%2 == 1 else 1) if self.ivty=='flip' else ((_coin(.5, 1) - .5) * 2)[0]
                self.Ws[iz][k] = self._A * mask
                self.ws[iz][k] = sgn * np.random.uniform(0.5, 1, self.M)
                self.bs[iz][k] = np.zeros(self.M) if k==0 else -sgn * np.random.uniform(5, 10, self.M)
                self.us[iz][k] = self.u if k==0 else self.u -(sgn * np.random.uniform(2, 5, self.M)) if not self.ivty=='shift'  else self.u -(sgn * np.random.uniform(5, 10, self.M))

    def _gen_noise(self):
        X = np.zeros((self.N, self.M))
        def _noise(loc, sc, sz):
            if self.noise == 'normal': Xi = np.random.normal(loc, sc, sz)
            elif self.noise == 'exp': Xi = np.random.exponential(sc, sz) + loc
            elif self.noise == 'gumbel': Xi = np.random.gumbel(loc, sc, sz)
            elif self.noise == 'unif': Xi = np.random.uniform(loc, sc, sz)
            else: Xi = np.zeros((sz))
            return Xi

        for i in self.G:
            # Non-sources
            confounded_source = any([i in conf_ind for conf_ind in self.conf_ind_sets])
            confounded_source = confounded_source and len(list(self.G.predecessors(i)))==0
            if not confounded_source:
                X[:, i] = _noise(0, self.NOISE_SC, self.N)
                continue
            # Source nodes
            for iz, conf_ind in enumerate(self.conf_ind_sets):
                if not (i in conf_ind):
                    continue
                for k in np.unique(self.Zs[iz]):
                    X[:, i][self.Zs[iz] == k] = _noise(self.bs[iz][k][i], self.NOISE_SC, len(X[:, i][self.Zs[iz] == k]))

                if self.vb > 0:
                    print(f"Mixing source {i},\tclass biases {[f'{self.bs[iz][k][i]:.2f}' for k in range(len(self.bs[iz]))]}")
        return X

    def plot_X(self, X, svfl=None, figsize=(7, 5)):

        for i in self.G:
            pa_i = list(self.G.predecessors(i))
            fig, axes = plt.subplots(len(pa_i) + 1, 1, figsize=figsize)

            true_labels = np.zeros(X.shape[0])
            for zi, node_set in enumerate(self.conf_ind_sets):
                if i in node_set:
                    true_labels = pi_join(true_labels, self.Zs[zi])
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
            if svfl is not None:
                os.makedirs(svfl, exist_ok=True)
                plt.savefig(svfl + f"Node_{i}_pa_{pa_i}_truth")
                plt.close()
            else:
                plt.show()


    def plot_X_idls(self, X, idls, svfl=None, method_idf='',figsize=(7, 5)):
        for i in self.G:
            pa_i = list(self.G.predecessors(i))
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
                plt.savefig(svfl + f"Node_{i}_pa_{pa_i}_estim_{method_idf}")
                plt.close()
            else:
                plt.show()



    def plot_X_pprobas(self, X, idls, pprobas, svfl=None, method_idf='', figsize=(7, 5)):
        for i in self.G:
            pa_i = list(self.G.predecessors(i))
            fig, axes = plt.subplots(len(pa_i) + 1, 1, figsize=figsize)

            labels = idls[i]
            confidences = np.max(pprobas[i], axis=1)

            ax1 = axes[0] if len(pa_i) > 0 else axes
            ax1.set_title(f"Node {i}: P(X{i})")
            cmap_i = plt.cm.get_cmap('tab10', len(np.unique(labels)))  # For class coloring

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
            else:
                plt.show()

