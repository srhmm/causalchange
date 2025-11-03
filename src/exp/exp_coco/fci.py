import numpy as np
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from src.coco.util.utils import data_to_jci, f1_score

#TODO st. from known mec
from src.exp.exp_coco.results_coco import MethodType
from src.coco.util.sparse_shift import dag2cpdag


class FCI_JCI():
    def __init__(self, D_obs, G, G_true, dag,
                 method : MethodType,
                 independence_test = 'fisherz',
                 true_dag_known = False):
        self.true_dag_known = true_dag_known
        self.n_c = D_obs.shape[0]
        self.n_nodes = D_obs.shape[2]
        self.nodes = G.nodes #without confounders
        self.independence_test = independence_test
        self.method = method

        D_aug, D_pooled, _ = data_to_jci(D_obs, 0, 0)
        self.D_obs = D_obs
        self.D_aug = D_aug
        self.D_pooled = D_pooled

        #Add a known MEC
        self.add_background_jci(G, G_true)
        self.add_background_pooled(G, G_true)
        #Results of FCI-JCI
        if method == MethodType.FCI_JCI:
            self._fci_jci(bg=True)
        elif method == MethodType.FCI_JCI_FULL:
            self._fci_jci(bg=False)
        #Results by pooling data
        elif method == MethodType.FCI_POOLED:
            self._fci_pooled(bg=True)
        elif method == MethodType.FCI_POOLED_FULL:
            self._fci_pooled(bg=False)
        #Results per context
        else:
            assert(method == MethodType.FCI_CONTEXT or method == MethodType.FCI_CONTEXT_FULL)
            bg = (method == MethodType.FCI_CONTEXT)
            self._fci_per_context(bg=bg)

        #print(self.eval_confounded(dag, method))
        #print(self.eval_causal(dag, method))

    def _fci_per_context(self, bg):
        self.G_c_bg = [None for _ in range(self.n_c)]
        self.G_c = [None for _ in range(self.n_c)]
        for ci, d in enumerate(self.D_obs):
            if bg:
                G, _ = fci(d, independence_test_method=self.independence_test,
                  background_knowledge=self.background_mec_pooled, verbose=False)
                self.G_c_bg[ci] = G
            else:
                G, _ = fci(d, independence_test_method=self.independence_test, verbose=False)
                self.G_c[ci] = G
            print(f"FCI_c{ci} (bg {bg}):")
            print(G.graph)
        print(f"FCI_cbest (bg {bg}):")

    def _fci_pooled(self, bg):
        if bg:
            G, _ = fci(self.D_pooled, independence_test_method=self.independence_test,
                  background_knowledge=self.background_mec_pooled, verbose=False)
            self.G_pooled_bg = G
        else:
            G, _ = fci(self.D_pooled, independence_test_method=self.independence_test,  verbose=False)
            self.G_pooled = G
        print(f"FCI_pooled (bg {bg}):")
        print(G.graph)

    def _fci_jci(self, bg):
        if bg:
            G, _ = fci(self.D_aug, independence_test_method=self.independence_test,
                  background_knowledge=self.background_mec, verbose=False)
            self.G_estim_bg = G
        else:
            G, _ = fci(self.D_aug, independence_test_method=self.independence_test, verbose=False)
            self.G_estim = G
        print(f"FCI_JCI (bg {bg}):")
        print(G.graph)

    def eval_causal(self, dag, method):
        return self._eval(self._eval_causal, dag, method)
    def eval_confounded(self, dag, method):
        return self._eval(self._eval_confounded, dag, method)

    def _eval(self, f, dag, method):
        if method == MethodType.FCI_JCI:
            return f(self.G_estim_bg, dag)
        elif method == MethodType.FCI_JCI_FULL:
            return f(self.G_estim, dag)
        elif method == MethodType.FCI_POOLED:
            return f(self.G_pooled_bg, dag)
        elif method == MethodType.FCI_POOLED_FULL:
            return f(self.G_pooled, dag)
        else:
            if method == MethodType.FCI_CONTEXT:
                Glist = self.G_c_bg
            else:
                assert(method == MethodType.FCI_CONTEXT_FULL)
                Glist = self.G_c
            best_res, F1 = None, 0
            for ci in range(self.n_c):
                res = f(Glist[ci], dag)
                f1 = res[4]
                if len(res)>5:#todo hacky
                    f1 = res[7]
                if f1 >= F1:
                    best_res, F1 = res, f1
            return best_res
    #could also eval. interventions/mechanism changes as adjacency to Ci variables.


    def add_background_pooled(self, G, G_true):
        nodes = []
        for i in G.nodes:#range(len(G.nodes)):
            node = GraphNode(f"X{i}")
            node.add_attribute("id", i)
            nodes.append(node)
        node_ind = max(G_true.nodes) #skip any indices reserved for the confounders.
        print("NODES:", [node.name for node in nodes], [node for node in G.nodes])

        # Convert the true DAG G to an adj (in top. order of the nodes, as in G.nodes) and then to a MEC, i.e. cpdag
        adj = np.array([np.zeros(len(G.nodes)) for _ in range(len(G.nodes))])
        for nj in G.nodes:
            for ni in G.predecessors(nj):

                    i = [n for n in G.nodes()].index(ni)#np.min(np.where(G.nodes == ni))
                    j = [n for n in G.nodes()].index(nj)# np.min(np.where(G.nodes == nj))

                    adj[i][j] = 1

        print("ADJ:")
        print(adj)
        if self.true_dag_known:
            true_cpdag = adj  #
        else:
            true_cpdag = dag2cpdag(adj)
            print("CPDAG:")
            print(true_cpdag)
        #For each fixed edge in the cpdag, add it to the background knowledge
        bg = BackgroundKnowledge()
        for i, ni in enumerate(G.nodes):
            for j in range(i+1, len(G.nodes)):#nj in G.nodes:
                if i == j:
                    continue
                #i = np.min(np.where(G.nodes == ni))
                #j = np.min(np.where(G.nodes == nj))
                # Look for directed edges
                if true_cpdag[i][j] == 1 and true_cpdag[j][i] == 0:
                    print(nodes[i].name, "->", nodes[j].name)
                    bg.add_required_by_node(nodes[i], nodes[j])
                elif true_cpdag[j][i] == 1 and true_cpdag[i][j] == 0:
                    print(nodes[j].name, "->", nodes[i].name)
                    bg.add_required_by_node(nodes[j], nodes[i])
                elif true_cpdag[j][i] == 0 and true_cpdag[i][j] == 0:
                    print(nodes[i].name, "x", nodes[j].name)
                    bg.add_forbidden_by_node(nodes[i], nodes[j])
                    bg.add_forbidden_by_node(nodes[j], nodes[i])

        self.background_mec_pooled = bg

    def add_background_jci(self, G, G_true):

        nodes = []
        for i in G.nodes:#range(len(G.nodes)):
            node = GraphNode(f"X{i}")
            node.add_attribute("id", i)
            nodes.append(node)
        node_ind = max(G_true.nodes) #skip any indices reserved for the confounders.
        print("NODES:", [node.name for node in nodes], [node for node in G.nodes])

        # JCI-FCI: Add a node corresp to each context
        for ci in range(self.n_c-1): #add one less, 4 context vars for 5 contexts
            node_ind = node_ind + 1
            node = GraphNode(f"X{node_ind}")
            node.add_attribute("id", node_ind)
            nodes.append(node)
        assert(len(nodes))==self.D_aug.shape[1]

        print("NODES:", [node.name for node in nodes],"\t", [node for node in G.nodes], ["C"+str(i) for i in range(self.n_c-1)])

        # Convert the true DAG G to an adj (in top. order of the nodes, as in G.nodes) and then to a MEC, i.e. cpdag
        adj = np.array([np.zeros(len(G.nodes)) for _ in range(len(G.nodes))])
        for nj in G.nodes:
            for ni in G.predecessors(nj):
                    i = [n for n in G.nodes()].index(ni)#np.min(np.where(G.nodes == ni))
                    j = [n for n in G.nodes()].index(nj)# np.min(np.where(G.nodes == nj))
                    adj[i][j] = 1

        print("ADJ:")
        print(adj)
        if self.true_dag_known:
            true_cpdag = adj  #
        else:
            true_cpdag = dag2cpdag(adj)
            print("CPDAG:")
            print(true_cpdag)

        #For each fixed edge in the cpdag, add it to the background knowledge
        bg = BackgroundKnowledge()
        for i, ni in enumerate(G.nodes):
            for j in range(i+1, len(G.nodes)):#nj in G.nodes:
                if i == j:
                    continue
                #i = np.min(np.where(G.nodes == ni))
                #j = np.min(np.where(G.nodes == nj))
                # Look for directed edges
                if true_cpdag[i][j] == 1 and true_cpdag[j][i] == 0:
                    print(nodes[i].name, "->", nodes[j].name)
                    bg.add_required_by_node(nodes[i], nodes[j])
                elif true_cpdag[j][i] == 1 and true_cpdag[i][j] == 0:
                    print(nodes[j].name, "->", nodes[i].name)
                    bg.add_required_by_node(nodes[j], nodes[i])
                #TODO necessary?
                #elif true_cpdag[j][i] == 0 and true_cpdag[i][j] == 0:
                #    print(nodes[i].name, "x", nodes[j].name)
                #    bg.add_forbidden_by_node(nodes[i], nodes[j])
                #    bg.add_forbidden_by_node(nodes[j], nodes[i])

        # For each context variable, add background knowledge that no nodes between context vars may exist,
         # and no edges towards a context var
        for ci in range(max(G_true.nodes), len(nodes)):
            print("context", ci)
            for cj in range(ci+1, len(nodes)):
                bg.add_forbidden_by_node(nodes[ci], nodes[cj])
                bg.add_forbidden_by_node(nodes[cj], nodes[ci])
                print(nodes[ci].name, "o", nodes[cj].name)
                print(nodes[cj].name, "o", nodes[ci].name)
            for ni in G.nodes:
                i = [n for n in G.nodes()].index(ni)  # np.min(np.where(G.nodes == ni))
                print(nodes[i].name, "o", nodes[ci].name)
                bg.add_forbidden_by_node(nodes[i], nodes[ci])
        self.background_mec = bg
    @staticmethod
    def _eval_confounded(G_estim, dag):
        nodes_confounded = dag.nodes_confounded
        G_obs_nodes = dag.G.nodes
        tp, fp, tn, fn = 0, 0, 0, 0
        tfp, ttp, ffp = 0, 0, 0

        adj = G_estim.graph
        for node_i in G_obs_nodes:
            for node_j in G_obs_nodes:

                i = [n for n in G_obs_nodes].index(node_i)
                j = [n for n in G_obs_nodes].index(node_j)

                cf = True in [(node_i in lst) and (node_j in lst) for lst in nodes_confounded]

                if adj[i][j] == 1 and adj[j][i] == 1:
                    if cf:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if cf:
                        fn += 1
                    else:
                        tn += 1

        return tp, fp, tn, fn, f1_score(tp, fp, fn)

    @staticmethod
    def _eval_causal(G_estim, dag): #G_obs, irrel_cf_nodes):
        G_obs = dag.G

        tp, fp, tn, fn = 0, 0, 0, 0

        adj = G_estim.graph
        for node_i in G_obs.nodes:
            for node_j in G_obs.nodes:
                #Iterate over nodes and not over indices in adj, cause those contain the context vars.
                i = np.min(np.where(G_obs.nodes == node_i))
                j = np.min(np.where(G_obs.nodes == node_j))

                causal = ((node_i, node_j) in G_obs.edges)

                if adj[i][j] == 1 and adj[j][i] == -1:
                    if causal:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if causal:
                        fn += 1
                    else:
                        tn += 1

        return tp, fp, tn, fn, f1_score(tp, fp, fn)

