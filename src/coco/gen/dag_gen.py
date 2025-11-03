from collections import defaultdict
import numpy as np
import networkx as nx
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import RBF

from statsmodels.compat import scipy

from src.coco.util.sparse_shift import cpdag2dags, dag2cpdag
from src.coco.util.utils import partition_to_map

def _random_nonlinearity():
    return np.random.choice([lambda x: x**2, np.sinc, np.tanh], 1)[0]

def _linearity():
    return lambda x: x

def _random_poly():
    return np.random.choice([lambda x: x**2, lambda x: x**3], 1)[0]

def kernel_rbf(X1, X2, length_scale=1):
    return RBF(length_scale=length_scale).__call__(X1, eval_gradient=False)

def kernel_exponentiated_quadratic(X1, X2):
    # as in https://peterroelants.github.io/posts/gaussian-process-tutorial/
    sq_norm = -0.5 * scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')
    return np.exp(sq_norm)

def _random_gp(X, n_functions, noise_scale=0.1,
               kernel_function=kernel_exponentiated_quadratic):
    # Independent variable samples
    #X = np.expand_dims(np.linspace(-4, 4, nb_of_samples), 1)
    if len(X.shape)==1:
        X = X.reshape(-1,1)
    E = kernel_exponentiated_quadratic(X, X)  # Kernel of data points

    # Draw samples from the prior at our data points.
    # Assume a mean of 0 for simplicity
    ys = np.random.multivariate_normal(
        mean=np.zeros(X.shape[0]), cov=E,
        size=n_functions) + np.random.normal(scale=noise_scale, size=X.shape[0])
    return ys

def _plot_random_gp(noise_scale=0.1):
    import matplotlib.pyplot as plt
    X = np.expand_dims(np.linspace(-4, 4, 500), 1)
    E = kernel_exponentiated_quadratic(X, X)
    ys = np.random.multivariate_normal(
        mean=np.zeros(X.shape[0]), cov=E,
        size=3) + np.random.normal(scale=noise_scale, size=X.shape[0])
    plt.scatter(X, ys[0])
    plt.scatter(X, ys[1])
    plt.scatter(X, ys[2])

def _random_gp_slow(X, kernel_function=kernel_rbf):
    if len(X.shape)==1:
        E = kernel_function(X.reshape(-1,1), X.reshape(-1,1))
    else:
        E = kernel_function(X, X)
    ys = np.random.multivariate_normal(
        mean=np.zeros(X.shape[0]), cov=E,
        size=1)
    y = ys[0, :] +np.random.normal(scale=0.2,size=X.shape[0])
    return y

def dag_to_amat(G, node_lookup):
    amat = np.zeros((len(G.nodes), len(G.nodes)))
    for n_i, n_j in G.edges:
        i = np.min(np.where(node_lookup==n_i))
        j = np.min(np.where(node_lookup==n_j))
        amat[i][j] = 1
    return amat
def dag_to_mec(G):
    node_lookup = [i for i in G.nodes]
    return cpdag2dags(dag2cpdag(np.array(dag_to_amat(G, node_lookup)))), node_lookup

def gen_data_from_graph(N, G, num_env, partitions, seed,
                        _functional_form = _random_nonlinearity(), #TODO _linearity(),
                        noise_iv=False, scale=False, reshape=True):
    X = np.zeros(shape=(num_env, len(G.nodes())+1, N))
    np.random.seed(seed)

    # Total: N * environments data points
    for i in G.nodes():

        X_ei = np.zeros(shape=(N))
        partition = partitions[i]
        np.random.seed(cantor_pairing(i, seed))
        B = _separable_coefficients(partition)

        for part_i, part in enumerate(partition):
            np.random.seed(cantor_pairing(cantor_pairing(i, seed), part_i))

            b_ji = B[part_i]
            f_ji = _functional_form
            sigma_i = np.random.uniform(1, 3)
            if noise_iv:
                mu_ji = np.random.uniform(0, 5)
            else:
                mu_ji = 0
            if np.random.uniform(0, 1) < .5:
                eps_i = np.random.normal(mu_ji, 1, N)
            else:
                eps_i = np.random.uniform(mu_ji+1, mu_ji+3, N)
            for e in part:
                X[e, i] += sigma_i*eps_i
                for j in G.predecessors(i):
                    X[e, i] += b_ji * f_ji(X[e, j])

    if scale:
        for c in range(len(X)):
            scaler = preprocessing.StandardScaler().fit(X[c])
            X[c] = scaler.transform(X[c])
    if reshape:
        X = np.array([X[c_i].T for c_i in range(len(X))])
    return X


def gen_random_directed_er(num_vars, seed, p=.3):
    indices = range(1, num_vars+1)
    np.random.seed(seed)
    topo_order = np.random.permutation(indices)
    G = nx.DiGraph([])
    edges = []
    for ind, i in enumerate(topo_order):
        for j in topo_order[ind+1:]:
            if np.random.RandomState(seed).uniform(0, 1) < p:
                edges.append((i, j))
    G.add_nodes_from(topo_order)
    G.add_edges_from(edges)
    return G

def gen_confounded_random_directed_er(is_bivariate,
        num_vars, num_confounders, num_confounded, seed, p=.3):
    if is_bivariate :
        return _gen_confounded_bivariate_directed_er(num_vars, num_confounders, num_confounded, seed)
    else:
        return _gen_confounded_random_directed_er(num_vars, num_confounders, num_confounded, seed, p)


def _gen_confounded_bivariate_directed_er(num_vars, num_confounders, num_confounded, seed):
    assert(num_confounders==1)
    assert(num_vars==2)
    indices = range(1, num_vars+2*num_confounders+1)
    np.random.seed(seed)
    topo_order = np.random.permutation(indices)
    #print(topo_order)
    G = nx.DiGraph([])
    edges = []

    G_observed = nx.DiGraph([])
    edges_observed = []
    topo_order_observed = [n for ind, n in enumerate(topo_order) if ind >= 2*num_confounders]

    if num_confounded[0] > 0:
        nodes_confounded = [topo_order_observed]
    else:
        nodes_confounded = []

    #print(nodes_confounded)
    for ind, i in enumerate(topo_order):
        if ind < num_confounders: #help node for each confounder
            j = topo_order[(ind+num_confounders)]
            edges.append((i, j))
            #print("Help Edge ", i ," -> ", j)
        elif num_confounders <= ind < 2* num_confounders:# actual confounders
            for j in topo_order[ind+1:]:
                cf_i = ind-num_confounders
                confounded = False
                if len(nodes_confounded)>0:
                    confounded = j in nodes_confounded[cf_i]
                if confounded:
                    edges.append((i, j))
                    #print("Confounded Edge ", i, " -> ", j)
        else:
            for j in topo_order[ind+1:]:
                    #print("Edge ", i ," -> ", j)
                edges.append((i, j))
                edges_observed.append((i, j))
    G.add_nodes_from(topo_order)
    G.add_edges_from(edges)


    G_observed.add_nodes_from(topo_order_observed)
    G_observed.add_edges_from(edges_observed)
    return G, G_observed, nodes_confounded

def _gen_confounded_random_directed_er(
        num_vars, num_confounders, num_confounded, seed, p=.3):
    indices = range(1, num_vars+2*num_confounders+1)
    np.random.seed(seed)
    topo_order = np.random.permutation(indices)
    #print(topo_order)
    G = nx.DiGraph([])
    edges = []

    G_observed = nx.DiGraph([])
    edges_observed = []
    topo_order_observed = [n for ind, n in enumerate(topo_order) if ind >= 2*num_confounders]
    #print(topo_order_observed)

    nodes_confounded = []
    remaining = [n for n in topo_order_observed]
    for cf_i in range(num_confounders):
        affected_nodes = np.random.choice(remaining, num_confounded[cf_i], replace=False)
        nodes_confounded.append(affected_nodes)
        remaining = [n for n in topo_order_observed if n not in nodes_confounded[cf_i]]

    #print(nodes_confounded)
    for ind, i in enumerate(topo_order):
        if ind < num_confounders: #help node for each confounder
            j = topo_order[(ind+num_confounders)]
            edges.append((i, j))
            #print("Help Edge ", i ," -> ", j)
        elif num_confounders <= ind < 2* num_confounders:# actual confounders
            for j in topo_order[ind+1:]:
                cf_i = ind-num_confounders
                confounded = j in nodes_confounded[cf_i]
                if confounded:
                    edges.append((i, j))
                    #print("Confounded Edge ", i, " -> ", j)
        else:
            for j in topo_order[ind+1:]:
                if np.random.uniform(0, 1) < p:
                    #print("Edge ", i ," -> ", j)
                    edges.append((i, j))
                    edges_observed.append((i, j))
    G.add_nodes_from(topo_order)
    G.add_edges_from(edges)


    G_observed.add_nodes_from(topo_order_observed)
    G_observed.add_edges_from(edges_observed)
    return G, G_observed, nodes_confounded


def gen_group_maps(num_env, partition_sizes, num_vars, seed, G=None):
    # Graph from which to generate data
    if G is None:
        G = gen_random_directed_er(num_vars, seed)
    # Each variable has its own partition of the environments
    partitions = defaultdict(list)
    for i in G.nodes:
        partition_size = np.random.choice(partition_sizes, 1)[0]
        partitions[i] = partition_to_map(gen_partition(seed, num_env, partition_size))
    return partitions


def gen_group_maps_nodeset(num_env, partition_sizes, num_vars, seed, nodes):

    # Each variable has its own partition of the environments
    partitions = defaultdict(list)
    for i in nodes:
        partition_size = np.random.choice(partition_sizes, 1)[0]
        partitions[i] = partition_to_map(gen_partition(seed, num_env, partition_size))
    return partitions

def gen_partitions(num_env, partition_size, num_vars, seed, G=None):
    # Graph from which to generate data
    if G is None:
        G = gen_random_directed_er(num_vars, seed)
    # Each variable has its own partition of the environments
    partitions = defaultdict(list)
    for i in G.nodes:
        # Random partition = Random permutation + Random split
        ##envs = np.random.RandomState(seed).permutation(list(range(num_env)))
        ##indices = np.sort(np.random.choice(range(1, num_env), partition_size, replace=False))
        ##partitions[i].append(list(envs[:indices[0]]))
        ##for k, index in enumerate(indices[0:-1]):
        ##    partitions[i].append(list(envs[index:indices[k+1]]))
        ##partitions[i].append(list(envs[indices[-1]:]))
        partitions[i] = gen_partition(seed, num_env, partition_size)
    return partitions

def gen_partition(seed, num_env, partition_size):
    #INTERVENED = True
    #if INTERVENED:
    #    partition = list()
    #    envs = np.random.RandomState(seed).permutation(list(range(num_env)))
    #    for i in range(partition_size):
    #        partition.append([envs[i]])
    #    partition.append([i for i in envs[partition_size:len(envs)]])
    #    return partition

    if partition_size==0:
        return [[i for i in range(num_env)]]
    if partition_size==num_env:
        raise ValueError(f'{partition_size} interventions for {num_env} contexts not allowed')
    partition = list()
    envs = np.random.RandomState(seed).permutation(list(range(num_env)))
    indices = np.sort(np.random.choice(range(1, num_env), partition_size, replace=False))
    partition.append(list(envs[:indices[0]]))
    for k, index in enumerate(indices[0:-1]):
        partition.append(list(envs[index:indices[k+1]]))
    partition.append(list(envs[indices[-1]:]))
    return partition



def _random_coefficients(partition):
    return [np.random.uniform(0.5, 2.5) for i in range(partition)]

def _separable_coefficients(partition, seed):
    np.random.seed(seed)
    b = np.random.uniform(0.5, 2.5)
    B = [b for _ in partition]
    for i in range(1, len(partition)):
        B[i] = B[i-1] + np.random.uniform(0.5, 2.5)
    #TODO rescale?
    return B



def cantor_pairing(x, y):
    seed = int((x + y) * (x + y + 1) / 2 + y)
    try:
        np.random.seed(seed)
    except ValueError:
        seed = int(100*np.random.uniform())
    return seed
