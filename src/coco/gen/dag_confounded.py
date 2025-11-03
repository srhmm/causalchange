from collections import defaultdict

import numpy as np
from sklearn import preprocessing
from src.coco.gen.dag_gen import gen_group_maps, gen_group_maps_nodeset, _random_nonlinearity, \
    cantor_pairing, _separable_coefficients, _random_gp, gen_confounded_random_directed_er
from src.coco.util.utils import confound_partition, map_to_partition


def sample_confounded_nodes(n_nodes, n_confounders, n_shifts_observed, n_shifts_confounders, exact_shifts=False):
    remaining = n_nodes - n_confounders * 2
    remaining_nodes = n_nodes
    if remaining < 0:
        raise ValueError('More confounders than node pairs')
    list_n_confounded_nodes = []
    for _ in range(n_confounders):
        if remaining <= 0:
            size = 2
        else:
            size = 2 + np.random.choice([i for i in range(remaining + 1)], 1)[0]
        list_n_confounded_nodes.append(size)
        remaining = remaining - size
        remaining_nodes = remaining_nodes - size
    list_n_shifts_observed = [0, n_shifts_observed]
    if exact_shifts:
        list_n_shifts_observed = [n_shifts_observed]
    list_n_shifts_confounders = [n_shifts_confounders]
    return list_n_confounded_nodes, list_n_shifts_confounders, list_n_shifts_observed


class DAGConfounded:
    def __init__(self, seed,
                 n_contexts,
                 n_observed_nodes, n_confounders,
                 n_shifts_observed, n_shifts_confounders,
                 #list_n_confounded_nodes, list_n_causal_mechanisms_per_node, list_n_causal_mechanisms_per_confounder,
                 is_bivariate,
                 from_adj=None
                 #test: CoCoTestType,
                 #sampler=None,
                 ):

        self.is_bivariate = is_bivariate
        np.random.seed(seed)
        self.seed = seed
        self.n_c = n_contexts
        self.n_nodes = n_observed_nodes
        self.n_confounders = n_confounders
        self.confounders = range(self.n_confounders)

        list_n_confounded_nodes, n_groups_confounder, n_groups\
            = sample_confounded_nodes(n_observed_nodes, n_confounders, n_shifts_observed, n_shifts_confounders)
        self.n_confounded = list_n_confounded_nodes
        self.n_groups = n_groups
        self.n_groups_confounder = n_groups_confounder

        self._gen_partitions()

        #self._oracle_similarities()

    #def node_index_confounder(self, cf_i):
    #    return len(self.G.nodes) + 1 + cf_i
    #def node_index_help_node(self, cf_i):
    #    return len(self.G.nodes) + 1 + self.n_confounders + cf_i

    #def confounder_index_node(self, i):
    #    return np.where(self.G_true.nodes==i)#i - len(self.G.nodes) -1

    def _gen_partitions(self):
        # Confounding
        assert(sum(self.n_confounded) <= self.n_nodes) # - for now, choose disjoint sets of affected nodes per confounder. later, allow overlaps.


        #G = gen_random_directed_er(self.n_nodes, self.seed)
        #G_true = gen_random_directed_er(self.n_nodes + 2 * self.n_confounders, self.seed)

        self.G_true, self.G, self.nodes_confounded = \
            gen_confounded_random_directed_er(self.is_bivariate, self.n_nodes, self.n_confounders, self.n_confounded, self.seed)

        self.edges = self.G.edges
        self.nodes = self.G.nodes
        self.maps_nodes_star = gen_group_maps(self.n_c, self.n_groups, self.n_nodes, self.seed, self.G)
        self.maps_confounders = gen_group_maps_nodeset(self.n_c, self.n_groups_confounder, self.n_confounders, self.seed, self.confounders)
        self.maps_nodes = defaultdict(list)

        #for cf_i in self.confounders:
        #    node_j = self.node_index_confounder(cf_i)
        #    self.G_true.add_node(node_j)
        #    help_node_j = self.node_index_help_node(cf_i)
        #    self.G_true.add_node(help_node_j)
        #    self.G_true.add_edge(help_node_j, node_j)

        # Update mechanism changes
        for node_j in self.nodes:
            affected = [node_j in self.nodes_confounded[cf_i] for cf_i in range(len(self.nodes_confounded))]
            confounded = True in affected

            if confounded:
                cf_i = np.min(np.where(affected))
                map_star = self.maps_nodes_star[node_j]
                map_cf = self.maps_confounders[cf_i]
                map_confounded = confound_partition(map_star, map_cf, self.n_c)
                self.maps_nodes[node_j] = map_confounded

                #self.G_true.add_edge(self.node_index_confounder(cf_i), node_j)
            else:
                self.maps_nodes[node_j] = self.maps_nodes_star[node_j]


    @staticmethod
    def show_edge_resids(D, nodes_X, node_y):

        from sklearn.linear_model import LinearRegression
        import matplotlib.pyplot as plt
        n_c = D.shape[0]
        n_samp = D.shape[1]
        X = D[:, :, nodes_X]
        y = D[:, :, node_y]

        X_k, y_k = X[0], y[0]
        model = LinearRegression().fit(X_k, y_k)
        for n_context in range(n_c):
            #model = LinearRegression().fit(X_k, y_k)
            X_k, y_k = X[n_context],y[n_context]
            N, p = X_k.shape[0], X_k.shape[1] + 1
            X_incpt = np.empty(shape=(N, p), dtype=np.float)
            X_incpt[:, 0] = 1
            X_incpt[:, 1:p] = X_k

            y_hat = model.predict(X_k)
            residuals = y_k - y_hat
            linsp = np.expand_dims(np.linspace(0, 100, n_samp), 1)
            plt.scatter(linsp, residuals)

    @staticmethod
    def show_edge(D, node_x, node_y):
        import matplotlib.pyplot as plt
        n_c = D.shape[0]
        X = D[:, :, node_x]
        y = D[:, :, node_y]
        for n_context in range( n_c):
            plt.scatter(X[n_context],y[n_context],  label='Context ' + str(n_context))
        plt.legend()
        plt.title('Edge ' + str(node_x) + ' -> ' + str(node_y))



    def gen_data(self, seed, n_samp,
                 _functional_form = _random_nonlinearity(),
                 oracle_partition=False,
                 noise_iv=False,
                 scale=False, reshape=True):

        n_observed_nodes = len(self.G.nodes())
        n_nodes = len(self.G_true.nodes()) + 1 #+1 for clear indexing.

        X = np.zeros(shape=(self.n_c, n_observed_nodes, n_samp))

        X_true = np.zeros(shape=(self.n_c, n_nodes, n_samp))

        np.random.seed(seed)
        index = [i for i in self.G.nodes] # at 0 of X, find node number index[0]

        # Total: N * environments data points
        for ind, i in enumerate(self.G_true.nodes()):

            is_confounder = False
            #    partition = map_to_partition(self.maps_nodes_star[i])
            if i in self.G.nodes:
                xi = ind - 2*self.n_confounders # +1
                #print('node', xi, index[xi])
                #By directly generating data from the confounded partition, avoid that the confounder has no effect:
                if oracle_partition:
                    partition = map_to_partition(self.maps_nodes[i])
                #Or use the original partition:
                else:
                    partition = map_to_partition(self.maps_nodes_star[i])
            else:
                xi = -1
                cf_i = ind - self.n_confounders
                #print('unobs', cf_i)
                if 0 <= cf_i < self.n_confounders:
                    is_confounder = True
                    if oracle_partition:
                        partition = [[c_i for c_i in range(self.n_c)]] #dont need mechanism changes as we directly generated data from the oracle partition
                    else:
                        #partition = [[c_i for c_i in range(self.n_c)]]
                        partition = map_to_partition(self.maps_confounders[cf_i])
                else: # help node: no mechanism change
                    #partition = map_to_partition(self.maps_confounders[(cf_i-self.n_confounders)])
                    #TODO in addition simulate a noise shift?
                    partition = [[c_i] for c_i in range(self.n_c)]


            B = _separable_coefficients(partition,cantor_pairing(i, seed))

            #TODOPRINT
            #print("True Edge", [j for j in self.G_true.predecessors(i)], "->", i, "\t", partition, "\tCoef.", [np.round(bi, 2) for bi in B])

            for part_i, part in enumerate(partition):
                np.random.seed(cantor_pairing(cantor_pairing(i, seed), part_i))

                b_ji = B[part_i]
                f_ji = _functional_form
                sigma_i = np.random.uniform(1, 3)

                if noise_iv or is_confounder:
                    sigma_i = np.random.uniform(5, 20)
                    mu_ji = np.random.uniform(max(0, (part_i-1)*20),max(20, (part_i-1)*50))
                    var_ji = np.random.uniform(max(0, (part_i-1)*100),max(10, (part_i-1)*200))
                else:
                    mu_ji = 0
                    var_ji = 1
                #only gauss.noise f now
                #if np.random.uniform(0, 1) < .5:
                eps_i = np.random.normal(mu_ji, var_ji, n_samp)
                #else:
                #    eps_i = np.random.uniform(mu_ji + 1, mu_ji + 3, n_samp)

                for e in part:

                    X_true[e, i] += sigma_i * eps_i
                    if i in self.G.nodes:
                         X[e, xi] += sigma_i * eps_i
                    #print("\tY_", i, "in env.", e, "(cfd:", not (i in self.G.nodes ), ")" )
                    for j in self.G_true.predecessors(i):
                        if is_confounder:
                            continue
                        #print("\t+=", b_ji, "* X_", j)

                        X_true[e, i] +=  b_ji * f_ji(X_true[e, j])
                        if i in self.G.nodes:
                            X[e, xi] +=b_ji * f_ji(X_true[e, j])

        if scale:
            for c in range(len(X_true)):
                #scaler = preprocessing.StandardScaler().fit(X[c])
                #X[c] = scaler.transform(X[c])
                scaler = preprocessing.StandardScaler().fit(X_true[c])
                X_true[c] = scaler.transform(X_true[c])
        if reshape:
            X = np.array([X[c_i].T for c_i in range(len(X))])
            X_true = np.array([X_true[c_i].T for c_i in range(len(X_true))])

        return X_true, X


    def gen_data_from_gp(self, seed, n_samp, _functional_form = _random_nonlinearity(),
                 noise_iv = False, scale = False, reshape = True):

        n_observed_nodes = len(self.G.nodes()) + 1
        n_nodes = len(self.G_true.nodes()) + 1 #+1 for clear indexing.

        X = np.zeros(shape=(self.n_c, n_observed_nodes, n_samp))
        X_true = np.zeros(shape=(self.n_c, n_nodes, n_samp))

        ys_gp = np.zeros(shape=(self.n_c, n_nodes, n_samp))
        np.random.seed(seed)

        # Total: N * environments data points
        for i in self.G_true.nodes():

            #X_ei = np.zeros(shape=(n_samp))

            #A test by directly generating data from the confounded partition:
            #if confounded_from_partition:
            #    partition = map_to_partition(self.maps_nodes[i])
            #else:
            #    partition = map_to_partition(self.maps_nodes_star[i])
            if i in self.G.nodes:
                partition = map_to_partition(self.maps_nodes_star[i])
            else:
                cf_i = self.confounder_index_node(i)
                if cf_i < self.n_confounders:
                    partition = map_to_partition(self.maps_confounders[cf_i])
                else: # help node: partition d
                    partition = [[c_i for c_i in range(self.n_c)]]

            B = _separable_coefficients(partition, i, seed)

            #TODOPRINT
            #print("True Edge", [j for j in self.G_true.predecessors(i)], "->", i, "\t", partition, "\tCoef.", [np.round(bi) for bi in B])

            for part_i, part in enumerate(partition):
                np.random.seed(cantor_pairing(cantor_pairing(i, seed), part_i))

                b_ji = B[part_i]
                f_ji = _functional_form
                sigma_i = np.random.uniform(1, 3)

                if noise_iv: # or len([i for i in self.G_true.predecessors(i)])==0:#special case pa(X)=[]: noise interv
                    mu_ji = np.random.uniform(0, 5)
                else:
                    mu_ji = 0
                if np.random.uniform(0, 1) < .5:
                    eps_i = np.random.normal(mu_ji, 1, n_samp)
                else:
                    eps_i = np.random.uniform(mu_ji + 1, mu_ji + 3, n_samp)

                for j in self.G_true.predecessors(i):
                    ys_gp[:, j] = _random_gp(X_true[:, j].reshape(-1,1), n_functions=1)[0].reshape(self.n_c, n_samp)

                for e in part:

                    ##B.X_true[e, i] += sigma_i * eps_i
                    ##B.if i in self.G.nodes:
                    ##B.     X[e, i] += sigma_i * eps_i
                    for j in self.G_true.predecessors(i):

                        X_true[e, i]  += ys_gp[e, j]
                        ##B. X_true[e, i] += b_ji * f_ji(X_true[e, j])
                        if i in self.G.nodes:
                            X[e, i] += ys_gp[e, j]
                            ##B. X[e, i] += b_ji * f_ji(X_true[e, j])

        if scale:
            for c in range(len(X)):
                scaler = preprocessing.StandardScaler().fit(X[c])
                X[c] = scaler.transform(X[c])
                scaler = preprocessing.StandardScaler().fit(X_true[c])
                X_true[c] = scaler.transform(X_true[c])
        if reshape:
            X = np.array([X[c_i].T for c_i in range(len(X))])
            X_true = np.array([X_true[c_i].T for c_i in range(len(X))])

        return X, X_true


def clus_sim_agglo(s, n):
    from sklearn.cluster import AgglomerativeClustering
    m = AgglomerativeClustering(
        metric='precomputed',
        n_clusters=n,
        linkage='complete'
    ).fit(s)
    return m.labels_

def clus_sim_spectral(s, n):
    from sklearn.cluster import SpectralClustering
    m = SpectralClustering(n_clusters=n,affinity='precomputed').fit(s)
    return m.labels_