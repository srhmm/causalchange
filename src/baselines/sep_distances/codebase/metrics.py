
#from codebase import mixed_graph as mixed
import networkx as nx
import numpy as np
import copy
import itertools
import scipy as sp


import gadjid

#-------------------SHD---------------------

#def SHD(graph1,graph2, normalized = True):
#    if not graph1.nnodes == graph2.nnodes:
#        raise ValueError('Graphs have different number of nodes')
#    else:
#        adj_1 = nx.adjacency_matrix(graph1.to_nx())
#        adj_2 = nx.adjacency_matrix(graph2.to_nx())
#        diff = np.abs(adj_1-adj_2)
#        diff = diff + diff.transpose()    #we do not want to count wrong orientations twice as strongly as missing edges
#        diff[diff > 1] = 1
#        total_nb_variables = adj_1.shape[0]
#        normalization = total_nb_variables*(total_nb_variables-1)/2
#        #print(normalization)
#        if normalized == True:
#            return np.sum(diff)/(2*normalization)
#        else:
#            return np.sum(diff)/2

def SHD_DAGs(graph1,graph2, normalized = True):
    if not graph1.nnodes == graph2.nnodes:
        raise ValueError('Graphs have different number of nodes')
    else:
        diff = 0
        node_list = list(graph1.nodes)
        for i in node_list:
            for j in node_list:
                if (i, j) in graph1._directed.keys() and (i, j) not in graph2._directed.keys():
                    diff += 1
                if (i, j) in graph2._directed.keys() and (i, j) not in graph1._directed.keys():
                    diff += 1

        N = graph1.nnodes

        normalization = N*(N-1)/2
        if normalized == True:
            return diff/normalization
        else:
            return diff

def SHD_CPDAGs(graph1,graph2, normalized = True):
    if not graph1.nnodes == graph2.nnodes:
        raise ValueError('Graphs have different number of nodes')
    else:
        diff = 0
        node_list = list(graph1.nodes)
        for i in node_list:
            for j in node_list:
                if (i, j) in graph1._directed.keys() and (i, j) not in graph2._directed.keys():
                    diff += 1
                if (i, j) in graph2._directed.keys() and (i, j) not in graph1._directed.keys():
                    diff += 1

        for i in node_list:
            for j in node_list[node_list.index(i)+1:]:
                if frozenset((i, j)) in graph1._undirected.keys() and frozenset((i, j)) not in graph2._undirected.keys() and (i,j) not in graph2._directed.keys() and (j,i) not in graph2._directed.keys(): #avoids double counts
                    diff += 1
                if frozenset((i, j)) in graph2._undirected.keys() and frozenset((i, j)) not in graph1._undirected.keys() and (i,j) not in graph1._directed.keys() and (j,i) not in graph1._directed.keys():  #avoids double counts
                    diff += 1

        N = graph1.nnodes

        normalization = N*(N-1)/2
        if normalized == True:
            return diff/normalization
        else:
            return diff

def SHD_MAGs(graph1,graph2, normalized = True):
    if not graph1.nnodes == graph2.nnodes:
        raise ValueError('Graphs have different number of nodes')
    else:
        diff = 0
        node_list = list(graph1.nodes)
        for i in node_list:
            for j in node_list:
                if (i, j) in graph1._directed.keys() and (i, j) not in graph2._directed.keys():
                    diff += 1
                if (i, j) in graph2._directed.keys() and (i, j) not in graph1._directed.keys():
                    diff += 1

        for i in node_list:
            for j in node_list[node_list.index(i)+1:]:
                if frozenset((i, j)) in graph1._bidirected.keys() and frozenset((i, j)) not in graph2._bidirected.keys() and (i, j) not in graph2._directed.keys() and (j, i) not in graph2._directed.keys():  # avoids double counts
                    diff += 1

                if frozenset((i, j)) in graph2._bidirected.keys() and frozenset((i, j)) not in graph1._bidirected.keys() and (i, j) not in graph1._directed.keys() and (j, i) not in graph1._directed.keys():  # avoids double counts
                    diff += 1


        N = graph1.nnodes

        normalization = N*(N-1)/2
        if normalized == True:
            return diff/normalization
        else:
            return diff

#---------------AIDs (based on the gadjid package) ------------------


def parent_AID_DAGs(graph1,graph2, normalized = True):
    if not graph1.nnodes == graph2.nnodes:
        raise ValueError('Graphs have different number of nodes')
    else:
        adj_1 = nx.to_numpy_array(graph1.to_nx(),dtype=np.int8)
        adj_2 = nx.to_numpy_array(graph2.to_nx(),dtype=np.int8)
        if normalized == True:
            return gadjid.parent_aid(adj_1,adj_2,edge_direction="from row to column")[0]
        else:
            return gadjid.parent_aid(adj_1,adj_2,edge_direction="from row to column")[1]

def parent_AID_CPDAGs(graph1,graph2, normalized = True):
    if not graph1.nnodes == graph2.nnodes:
        raise ValueError('Graphs have different number of nodes')
    else:
        N = graph1.nnodes

        node_list = list(graph1.nodes)

        adj_1 = np.zeros((N,N),dtype=np.int8)
        adj_2 = np.zeros((N,N),dtype=np.int8)

        for (i,j) in graph1._directed.keys():
            adj_1[node_list.index(i),node_list.index(j)] = 1

        for (i,j) in graph1._undirected.keys():
            adj_1[node_list.index(i),node_list.index(j)] = 2
            adj_1[node_list.index(j), node_list.index(i)] = 2

        for (i, j) in graph2._directed.keys():
            adj_2[node_list.index(i), node_list.index(j)] = 1

        for (i, j) in graph2._undirected.keys():
            adj_2[node_list.index(i), node_list.index(j)] = 2
            adj_2[node_list.index(j), node_list.index(i)] = 2

        if normalized == True:
            #print('Adj1', adj_1)
            #print('Adj2',adj_2)
            return gadjid.parent_aid(adj_1,adj_2,edge_direction="from row to column")[0]
        else:
            return gadjid.parent_aid(adj_1,adj_2,edge_direction="from row to column")[1]

def ancestor_AID_DAGs(graph1,graph2, normalized = True):
    if not graph1.nnodes == graph2.nnodes:
        raise ValueError('Graphs have different number of nodes')
    else:
        adj_1 = nx.to_numpy_array(graph1.to_nx(),dtype=np.int8)
        adj_2 = nx.to_numpy_array(graph2.to_nx(),dtype=np.int8)
        if normalized == True:
            return gadjid.ancestor_aid(adj_1,adj_2,edge_direction="from row to column")[0]
        else:
            return gadjid.ancestor_aid(adj_1, adj_2,edge_direction="from row to column")[1]

def ancestor_AID_CPDAGs(graph1,graph2, normalized = True):
    if not graph1.nnodes == graph2.nnodes:
        raise ValueError('Graphs have different number of nodes')
    else:
        N = graph1.nnodes

        node_list = list(graph1.nodes)

        adj_1 = np.zeros((N,N),dtype=np.int8)
        adj_2 = np.zeros((N,N),dtype=np.int8)

        for (i,j) in graph1._directed.keys():
            adj_1[node_list.index(i),node_list.index(j)] = 1

        for (i,j) in graph1._undirected.keys():
            adj_1[node_list.index(i),node_list.index(j)] = 2
            adj_1[node_list.index(j), node_list.index(i)] = 2

        for (i, j) in graph2._directed.keys():
            adj_2[node_list.index(i), node_list.index(j)] = 1

        for (i, j) in graph2._undirected.keys():
            adj_2[node_list.index(i), node_list.index(j)] = 2
            adj_2[node_list.index(j), node_list.index(i)] = 2

        if normalized == True:
            return gadjid.ancestor_aid(adj_1,adj_2,edge_direction="from row to column")[0]
        else:
            return gadjid.ancestor_aid(adj_1,adj_2,edge_direction="from row to column")[1]

def sym_parent_AID_DAGs(graph1,graph2,normalized= True ):
    '''Computes the symmetrized AID, see function parent_AID_DAGs() for input specifications'''
    return 0.5*parent_AID_DAGs(graph1,graph2,normalized) + 0.5*parent_AID_DAGs(graph2,graph1,normalized)

def sym_parent_AID_CPDAGs(graph1,graph2,normalized= True ):
    '''Computes the symmetrized SD, see function parent_AID_CPDAGs() for input specifications'''
    return 0.5*parent_AID_CPDAGs(graph1,graph2,normalized) + 0.5*parent_AID_CPDAGs(graph2,graph1,normalized)

def sym_ancestor_AID_DAGs(graph1,graph2,normalized= True ):
    '''Computes the symmetrized AID, see function ancestor_AID_DAGs() for input specifications'''
    return 0.5*ancestor_AID_DAGs(graph1,graph2,normalized) + 0.5*ancestor_AID_DAGs(graph2,graph1,normalized)

def sym_ancestor_AID_CPDAGs(graph1,graph2,normalized= True ):
    '''Computes the symmetrized SD, see function ancestor_AID_CPDAGs() for input specifications'''
    return 0.5*ancestor_AID_CPDAGs(graph1,graph2,normalized) + 0.5*ancestor_AID_CPDAGs(graph2,graph1,normalized)

#separation distances

def SD_DAGs(graph1,graph2, type= 'parent', normalized = True, MB_enhanced = False):

    '''Input:
            graph1, graph2: DAGs implemented in the LabelledMixedGraph class, see mixed_graph.py
            type: string, either 'parent', 'ancestor', or 'ZL'
            normalized: Boolean
            MB_enhanced: Boolean
     '''
    G_1 = graph1
    G_2 = graph2

    if G_1.nodes != G_2.nodes:
        raise ValueError('graphs have different nodes!')

    variables = G_1.nodes
    N = len(variables)

    if MB_enhanced is False:

        separable_node_pairs = {}
        ''' Get union of parents '''
        if type == 'parent':
            for (node1,node2) in itertools.combinations(G_2.nodes,r=2):
                if not ((node1,node2) in G_2.directed.keys() or (node2,node1) in G_2.directed.keys()):
                    separable_node_pairs[(node1,node2)] = set(G_2.parents_of(node1)).union(set(G_2.parents_of(node2)))

        ''' Get union of ancestors '''
        if type == 'ancestor':
            for (node1, node2) in itertools.combinations(G_2.nodes, r=2):
                if not ((node1, node2) in G_2.directed.keys() or (node2,node1) in G_2.directed.keys()):
                    anc1 = G_2.get_ancestors(node1)
                    anc2 = G_2.get_ancestors(node2)
                    if anc1 is not None:
                        anc1 = set(anc1).discard(node2)

                    if anc1 is None:
                        anc1 = set()
                    if anc2 is not None:
                        anc2 = set(anc2).discard(node1)
                    if anc2 is None:
                        anc2 = set()

                    separable_node_pairs[(node1,node2)] = anc1.union(anc2)

        if type in ['parent', 'ancestor']:
            error_count = 0

            for (X, Y) in separable_node_pairs.keys():

                # print({X},{Y},separable_node_pairs[(X,Y)])
                if not G_1.is_d_separated({X}, {Y}, set(separable_node_pairs[(X, Y)]),DAG_check = False):
                 #   print({X}, {Y}, separable_node_pairs[(X, Y)])
                    error_count += 1

            if normalized == True:
                return 2 * error_count / (N * (N - 1))
            else:
                return error_count

        if type == 'ZL':
            '''computation of the van der Zander-Liskiewicz separation distance.'''

            edges_2 = G_2.directed_keys

            pairs = list(itertools.product(G_2.nodes,G_2.nodes))
            for node in G_2.nodes:
                pairs.remove((node,node))

            for (node1,node2) in pairs:
                if not ((node1,node2) in edges_2 or (node2,node1) in edges_2):
                   separable_node_pairs[(node1, node2)] = G_2.find_minimal_d_separator({node1},{node2},DAG_check=False)

            error_count = 0
            #print(separable_node_pairs)

            for (X,Y) in separable_node_pairs.keys():

                #print({X},{Y},separable_node_pairs[(X,Y)])
                if not G_1.is_d_separated({X},{Y},set(separable_node_pairs[(X, Y)]),DAG_check = False):
                    error_count += 1

            if normalized == True:
                return error_count/(N*(N-1))
            else:
                return error_count

    elif MB_enhanced is True:

        '''Implementation of Markov enhancement'''

        error_count = 0

        G_2.get_Markov_blankets()

        for node in G_2.nodes:
            non_blanket_nodes = G_2.nodes - G_2.Markov_blankets[node] - {node}
            not_separated_in_G_1 = G_1.nodes - G_1.BayesBall({node},G_2.Markov_blankets[node]) - {node} - G_2.Markov_blankets[node]
            intersection = non_blanket_nodes.intersection(not_separated_in_G_1)
            #print(node,G_2.Markov_blankets[node], non_blanket_nodes, not_separated_in_G_1, intersection)
            error_count += len(intersection)
            adjacent_nodes_in_G_2 = G_2.adjacent_to(node)
            exceptions = G_2.Markov_blankets[node] - adjacent_nodes_in_G_2
            #print(exceptions)
            parents_of_node = set(G_2.parents_of(node))
            for exception_node in exceptions:
                if type == 'parent':
                    exceptional_sep_set = parents_of_node.union(set(G_2.parents_of(exception_node)))

                elif type == 'ZL':
                    exceptional_sep_set = set(G_2.find_minimal_d_separator({node},{exception_node},restricted=G_2.nodes, DAG_check = False)
)
                else:
                    raise ValueError('Markov enhancement is currently only implemented for parent separation and ZL separation on DAGs')

                if not G_1.is_d_separated({node}, {exception_node}, exceptional_sep_set,DAG_check = False):
                    error_count += 1

        if normalized == True:
            return error_count / (N * (N - 1))
        else:
            return error_count

def sym_SD_DAGs(graph1,graph2, type= 'parent',normalized= True, MB_enhanced = False):
    '''Computes the symmetrized SD, see function SD for input specifications'''
    return 0.5*SD_DAGs(graph1,graph2,type,normalized, MB_enhanced) + 0.5*SD_DAGs(graph2,graph1,type,normalized, MB_enhanced)

def weighted_SD_DAGs(graph1,graph2,weight, type= 'parent',normalized = True, MB_enhanced=False):
    '''Compute a weighted version of the SD, weight is the weight assigned to SD(graph1,graph2)'''
    return weight*SD_DAGs(graph1,graph2,type,normalized, MB_enhanced) + (1-weight)*SD_DAGs(graph2,graph1,type,normalized, MB_enhanced)

def SD_mixed_graphs(graph1,graph2, type= 'ZL', normalized = True, MB_enhanced = False):
    '''
    Computation of separation distance in acyclic mixed graphs.
    If your graphs are DAGs, SD_DAGs() offers more features.
    Input:
            graph1, graph2: DAGs implemented in the LabelledMixedGraph class, see mixed_graph.py
            type: string, only option is 'ZL' at this moment
            normalized: Boolean
            MB_enhanced: Boolean, however the True option is not yet implemented.
     '''

    G_1 = graph1
    G_2 = graph2


    if G_1.nodes != G_2.nodes:
        raise ValueError('graphs have different nodes!')

    variables = G_1.nodes
    N = len(variables)

    if MB_enhanced is False:

        separable_node_pairs = {}

        if type == 'ZL':
            '''computation of the van der Zander-Liskiewicz separation distance for mixed graphs'''
            canonical_DAG_1 = G_1.get_canonical_directed_graph()

            canonical_DAG_2 = G_2.get_canonical_directed_graph()

            pairs = list(itertools.product(G_2.nodes, G_2.nodes))
            for node in G_2.nodes:
                pairs.remove((node, node))

            for (node1, node2) in pairs:
                if not ((node1, node2) in G_2.directed_keys or (node2, node1) in G_2.directed_keys or frozenset(
                        {node2, node1}) in G_2.bidirected_keys):

                    sep = canonical_DAG_2.find_minimal_d_separator({node1}, {node2}, restricted=G_2.nodes,
                                                                   DAG_check=False)
                    if not sep is None:
                        separable_node_pairs[(node1, node2)] = sep
            #print(separable_node_pairs)
            error_count = 0

            for (X, Y) in separable_node_pairs.keys():

                if not canonical_DAG_1.is_d_separated({X}, {Y}, set(separable_node_pairs[(X, Y)]), DAG_check=False):
                    error_count += 1

            if normalized == True:
                return error_count / (N * (N - 1))
            else:
                return error_count


    elif MB_enhanced is True:

        raise ValueError('Markov enhancement is currently only implemented for parent and ZL separation on DAGs')


def sym_SD_mixed_graphs(graph1,graph2, type= 'ZL',normalized= True, MB_enhanced = False):
    '''Computes the symmetrized SD, see function SD for input specifications'''
    return 0.5*SD_mixed_graphs(graph1,graph2,type,normalized, MB_enhanced) + 0.5*SD_mixed_graphs(graph2,graph1,type,normalized, MB_enhanced)

def weighted_SD_mixed_graphs(graph1,graph2,weight, type= 'ZL',normalized = True, MB_enhanced=False):
    '''Compute a weighted version of the SD, weight is the weight assigned to SD(graph1,graph2)'''
    return weight*SD_mixed_graphs(graph1,graph2,type,normalized, MB_enhanced) + (1-weight)*SD_mixed_graphs(graph2,graph1,type,normalized, MB_enhanced)

def SD_CPDAGs(graph1,graph2,type='pparent',normalized = True, MB_enhanced = False):

    '''Input:
            graph1, graph2: CPDAGs implemented in the LabelledMixedGraph class, see mixed_graph.py
            type: string, either 'pparent', 'pancestor', or 'ZL'
            normalized: Boolean
            MB_enhanced: Boolean
     '''


    G_1 = graph1
    G_2 = graph2


    if G_1.nodes != G_2.nodes:
        raise ValueError('graphs have different nodes!')

    variables = G_1.nodes
    N = len(variables)

    if MB_enhanced is False:

        separable_node_pairs = {}
        ''' Get union of pparents '''

        if G_1._representative is None:
            G_1.get_representative_of_MEC()

        G_1_rep = G_1._representative

        if type == 'ZL':

            if G_2._representative is None:
                G_2.get_representative_of_MEC()
            G_2_rep = G_2._representative

            return SD_DAGs(G_1_rep, G_2_rep, type='ZL', normalized=normalized)

        if type == 'pparent':


            if (G_1.num_bidirected > 0 or G_1.num_semidirected > 0):
                raise ValueError(
                    'Graph1 has bidirected or semidirected edges.  This function is for CPDAGs.')

            if (G_2.num_bidirected > 0 or G_2.num_semidirected > 0):
                raise ValueError(
                    'Graph2 has bidirected or semidirected edges. This function is for CPDAGs.')


            for (node1, node2) in itertools.combinations(G_2.nodes, r=2):
                if not ((node1, node2) in G_2._directed or (node2, node1) in G_2._directed or frozenset({node1, node2}) in G_2._undirected):
                    separable_node_pairs[(node1, node2)] = set(G_2.possibleparents_of(node1)).union(set(G_2.possibleparents_of(node2))) -{node1} -{node2}
                    #print(node1, node2, separable_node_pairs[(node1, node2)])

        if type == 'pancestor':

            if (G_1.num_bidirected > 0 or G_1.num_semidirected > 0):
                raise ValueError(
                    'Graph1 has bidirected or semidirected edges.  This function is for CPDAGs.')

            if (G_2.num_bidirected > 0 or G_2.num_semidirected > 0):
                raise ValueError(
                    'Graph2 has bidirected or semidirected edges. This function is for CPDAGs.')


            for (node1, node2) in itertools.combinations(G_2.nodes, r=2):
                if not ((node1, node2) in G_2.directed.keys() or (node2, node1) in G_2.directed.keys() or frozenset({node1, node2}) in G_2._undirected or frozenset({node1, node2})in G_2._undirected):
                    anc1 = G_2.get_possible_ancestors(node1)
                    anc2 = G_2.get_possible_ancestors(node2)
                    if anc1 is not None:
                        anc1 = set(anc1).discard(node2)

                    if anc1 is None:
                        anc1 = set()
                    if anc2 is not None:
                        anc2 = set(anc2).discard(node1)
                    if anc2 is None:
                        anc2 = set()

                    separable_node_pairs[(node1, node2)] = anc1.union(anc2)


        error_count = 0
        #print(separable_node_pairs)
        for (X, Y) in separable_node_pairs.keys():

            #print({X},{Y},separable_node_pairs[(X,Y)])
            if not G_1_rep.is_d_separated({X}, {Y}, set(separable_node_pairs[(X, Y)])):
                #print('added')
                error_count += 1

        if normalized == True:
            return 2 * error_count / (N * (N - 1))
        else:
            return error_count

    elif MB_enhanced is True:

        if (G_1.num_bidirected > 0 or G_1.num_semidirected > 0):
            raise ValueError(
                'Graph1 has bidirected or semidirected edges. This function is for CPDAGs.')

        if (G_2.num_bidirected > 0 or G_2.num_semidirected > 0):
            raise ValueError(
                'Graph2 has bidirected or semidirected edges. This function is for CPDAGs.')

        if G_2._representative is None:
            G_2.get_representative_of_MEC()

        G_2_rep = G_2._representative
        G_2_rep.get_Markov_blankets()

        if G_1._representative is None:
            G_1.get_representative_of_MEC()

        G_1_rep = G_1._representative
        #G_1_rep_nx = G_1._representative.to_nx()

        error_count = 0
        for node in G_2_rep.nodes:
            non_blanket_nodes = G_2_rep.nodes - G_2_rep.Markov_blankets[node] - {node}
            not_separated_in_G_1 = G_1.nodes - G_1_rep.BayesBall({node}, G_2_rep.Markov_blankets[node]) - {node} - G_2_rep.Markov_blankets[node]
            intersection = non_blanket_nodes.intersection(not_separated_in_G_1)
            #print(node,G_2_rep.Markov_blankets[node], non_blanket_nodes,not_separated_in_G_1, intersection)
            error_count += len(intersection)
            adjacent_nodes_in_G_2 = G_2_rep.adjacent_to(node)
            exceptions = G_2_rep.Markov_blankets[node] - adjacent_nodes_in_G_2

            pos_parents_of_node = set(G_2.possibleparents_of(node))
            for exception_node in exceptions:
                if type == 'pparent':
                    exceptional_sep_set = pos_parents_of_node.union(set(G_2.possibleparents_of(exception_node))) - {node} - {exception_node}
                    #print('exception', node, exception_node,exceptional_sep_set)

                elif type == 'ZL':
                    exceptional_sep_set = set(G_2_rep.find_minimal_d_separator({node}, {exception_node}, restricted=G_2.nodes, DAG_check=False))
                else:
                    raise ValueError(
                        'Markov enhancement is currently only implemented for possible parent and ZL-separation')

                if not G_1_rep.is_d_separated({node}, {exception_node}, exceptional_sep_set):
                    #print(' exception: +1')
                    error_count += 1

        if normalized == True:
            return error_count / (N * (N - 1))
        else:
            return error_count


def sym_SD_CPDAGs(graph1,graph2, type= 'pparent',normalized= True, MB_enhanced = False):
    '''Computes the symmetrized SD, see function SD for input specifications'''
    return 0.5*SD_CPDAGs(graph1,graph2,type,normalized, MB_enhanced) + 0.5*SD_CPDAGs(graph2,graph1,type,normalized, MB_enhanced)


#------------------s/c-metrics-----------------

def generate_triples(nodes, order=0, all=True, max_number=1, random_state=None):

    if random_state == None:
        random_state = np.random
    triple_list = []
    variables = list(nodes)
    if all == True:
        if order == 0:
            for X in variables:
                helper_variables = copy.deepcopy(list(variables))
                helper_variables.remove(X)
                for Y in helper_variables:
                    if ((X,), (Y,), ()) not in triple_list and ((Y,), (X,), ()) not in triple_list:
                        triple_list.append(((X,), (Y,), ()))
        if order > 0:
            tuple_list = []
            for X in variables:
                helper_variables = copy.deepcopy(list(variables))
                helper_variables.remove(X)
                for Y in helper_variables:
                    if (X, Y) not in tuple_list and (Y, X) not in tuple_list:
                        # print((X,Y))
                        tuple_list.append((X, Y))
                        more_helper_variables = copy.deepcopy(helper_variables)
                        more_helper_variables.remove(Y)
                        cond_sets = list(itertools.combinations(more_helper_variables, order))
                        # print(cond_sets)
                        for sep_set in cond_sets:
                            triple_list.append(((X,), (Y,), sep_set))
        return triple_list
    else:
        N = len(variables)
        upper_bound = sp.special.binom(N, 2) * sp.special.binom(N - 2, order)
        if order > 0:
            while len(triple_list) < min(max_number, upper_bound):
                indices = range(len(variables))
                sep_pair = random_state.choice(indices, 2, replace=False)
                X = variables[sep_pair[0]]
                Y = variables[sep_pair[1]]
                helper_variables = copy.deepcopy(list(variables))
                helper_variables.remove(X)
                helper_variables.remove(Y)
                indices = range(len(helper_variables))
                sep_set_indices = random_state.choice(indices, order, replace=False)
                sep_set_indices.sort()
                sep_set = []
                for index in sep_set_indices:
                    sep_set.append(helper_variables[index])
                sep_set_new = tuple(sep_set)
                if ((X,), (Y,), sep_set_new) not in triple_list and ((Y,), (X,), sep_set_new) not in triple_list:
                    triple_list.append(((X,), (Y,), sep_set_new))

            return triple_list

        elif order == 0:
            while len(triple_list) < min(max_number, upper_bound):
                indices = range(len(variables))
                sep_pair = random_state.choice(indices, 2, replace=False)
                X = variables[sep_pair[0]]
                Y = variables[sep_pair[1]]
                if ((X,), (Y,), ()) not in triple_list and ((Y,), (X,), ()) not in triple_list:
                    triple_list.append(((X,), (Y,), ()))
            return triple_list

def metric_ungraded_DAGs(graph1, graph2, statements_to_be_compared, type ='sc'):


    distance = 0.0

    if type == 'sc':
        for test_triple in statements_to_be_compared:
            oracle1_result = graph1.is_d_separated(x=set(test_triple[0]), y=set(test_triple[1]), z=set(test_triple[2]))
            oracle2_result = graph2.is_d_separated(x=set(test_triple[0]), y=set(test_triple[1]), z=set(test_triple[2]))
            if oracle1_result == True and oracle2_result == False:
                distance +=1.
            if oracle1_result == False and oracle2_result == True:
                distance +=1.

        distance /= len(statements_to_be_compared)

        return distance

    elif type == 'Markov' or type == 'c':
        counter = 0
        for test_triple in statements_to_be_compared:
            oracle1_result = graph1.is_d_separated(x=set(test_triple[0]), y=set(test_triple[1]), z=set(test_triple[2]))
            if oracle1_result == False:
                counter += 1
                oracle2_result = graph2.is_d_separated(x=set(test_triple[0]), y=set(test_triple[1]), z=set(test_triple[2]))
                if oracle2_result == True:
                    distance += 1.
        if counter == 0:
            return distance
        else:
            distance /= counter
            return distance

    elif type == 'Faithfulness' or type == 's':
        counter = 0
        for test_triple in statements_to_be_compared:
            oracle1_result = graph1.is_d_separated(x=set(test_triple[0]), y=set(test_triple[1]), z=set(test_triple[2]))
            if oracle1_result == True:
                counter += 1
                oracle2_result = graph2.is_d_separated(x=set(test_triple[0]), y=set(test_triple[1]), z=set(test_triple[2]))
                if oracle2_result == False:
                    distance += 1.
        if counter == 0:
            return distance
        else:
            distance /= counter
            return distance

def metric_DAGs(graph1, graph2, type ='sc', max_order=None, randomize_higher_order = 0, normalized= True, random_state = None, include_distance_dict = False):
    '''implementation of s/c,s- and c-metric for directed acyclic graphs including random approximations.
       Input:

       graph1, graph2: directed acyclic graphs in the LabelledMixedGraph() format;
       type: options: string, 'sc', 'c' (alternative 'Markov') , 's' (alternative: 'Faithfulness');
       max_order: int, maximal order up to which the metric is computed;
       randomize_higher_order: int, if 0 no randomization, otherwise the number indicates the number of randomly chosen separation statements checked per order > max_order;
       normalized: Boolean,
       random_state: numpy RandomState object for randomization;
       include_distance_dict: Boolean, if True, dictionary of checked statements and result is returned.

    '''


    if graph1.nodes != graph2.nodes:
        raise ValueError('graphs have different nodes!')
    else:
        variables = graph1.nodes
        N = len(variables)

        if max_order is None:
            max_order = N-2

        distance = 0.0
        distance_dict = {}
        triple_list = []

        for k in range(max_order+1):
            k_triples = generate_triples(variables,order=k,all=True)
            if k_triples != []:
                distance_dict[k] = metric_ungraded_DAGs(graph1, graph2, statements_to_be_compared= k_triples, type= type)
                triple_list.extend(k_triples)
                distance += distance_dict[k]


        if randomize_higher_order > 0:

            if random_state == None:
                random_state = np.random

            for k in range(max_order+1,N-1):
                k_triples = generate_triples(variables, order=k, all=False, max_number=randomize_higher_order)
                distance_dict[k] = metric_ungraded_DAGs(graph1, graph2, statements_to_be_compared= k_triples, type= type)
                triple_list.extend(k_triples)
                distance += distance_dict[k]

            if normalized == True:
                distance /= (N-1)
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict
            else:
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict

        else:
            if normalized == True:
                distance /= max_order+1
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict
            else:
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict


def metric_CPDAGs(graph1, graph2, type ='sc', max_order=None, randomize_higher_order = 0, normalized= True, random_state = None, include_distance_dict = False):
    '''implementation of s/c,s- and c-metric for CPDAGs, including random approximations.
       Input:

       graph1, graph2: directed acyclic graphs in the LabelledMixedGraph() format;
       type: options: string, 'sc', 'c' (alternative 'Markov') , 's' (alternative: 'Faithfulness');
       max_order: int, maximal order up to which the metric is computed;
       randomize_higher_order: int, if 0 no randomization, otherwise the number indicates the number of randomly chosen separation statements checked per order > max_order;
       normalized: Boolean,
       random_state: numpy RandomState object for randomization;
       include_distance_dict: Boolean, if True, dictionary of checked statements and result is returned.

    '''

    if graph1._representative is None:
        graph1.get_representative_of_MEC()

    G_1 = graph1._representative

    if graph2._representative is None:
        graph2.get_representative_of_MEC()

    G_2 = graph2._representative

    if G_1.nodes != G_2.nodes:
        raise ValueError('graphs have different nodes!')
    else:
        variables = G_1.nodes
        N = len(variables)

        if max_order is None:
            max_order = N-2

        distance = 0.0
        distance_dict = {}
        triple_list = []

        for k in range(max_order+1):
            k_triples = generate_triples(variables,order=k,all=True)
            if k_triples != []:
                distance_dict[k] = metric_ungraded_DAGs(G_1, G_2, statements_to_be_compared= k_triples, type= type)
                triple_list.extend(k_triples)
                distance += distance_dict[k]


        if randomize_higher_order > 0:

            if random_state == None:
                random_state = np.random

            for k in range(max_order+1,N-1):
                k_triples = generate_triples(variables, order=k, all=False, max_number=randomize_higher_order)
                distance_dict[k] = metric_ungraded_DAGs(G_1, G_2, statements_to_be_compared= k_triples, type= type)
                triple_list.extend(k_triples)
                distance += distance_dict[k]

            if normalized == True:
                distance /= (N-1)
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict
            else:
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict

        else:
            if normalized == True:
                distance /= max_order+1
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict
            else:
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict

def metric_mixed_graphs(graph1, graph2, type ='sc', max_order=None, randomize_higher_order = 0, normalized= True, random_state = None, include_distance_dict = False):
    '''implementation of s/c,s- and c-metric for mixed graphs including random approximations.
       Input:

       graph1, graph2: directed acyclic graphs in the LabelledMixedGraph() format;
       type: options: string, 'sc', 'c' (alternative 'Markov') , 's' (alternative: 'Faithfulness');
       max_order: int, maximal order up to which the metric is computed;
       randomize_higher_order: int, if 0 no randomization, otherwise the number indicates the number of randomly chosen separation statements checked per order > max_order;
       normalized: Boolean,
       random_state: numpy RandomState object for randomization;
       include_distance_dict: Boolean, if True, dictionary of checked statements and result is returned.

    '''

    if graph1.nodes != graph2.nodes:
        raise ValueError('graphs have different nodes!')

    else:

        variables = graph1.nodes
        N = len(variables)

        G_1 = graph1.get_canonical_directed_graph()

        G_2 = graph2.get_canonical_directed_graph()



        if max_order is None:
            max_order = N-2

        distance = 0.0
        distance_dict = {}
        triple_list = []

        for k in range(max_order+1):
            k_triples = generate_triples(variables,order=k,all=True)
            if k_triples != []:
                distance_dict[k] = metric_ungraded_DAGs(G_1, G_2, statements_to_be_compared= k_triples, type= type)
                triple_list.extend(k_triples)
                distance += distance_dict[k]


        if randomize_higher_order > 0:

            if random_state == None:
                random_state = np.random

            for k in range(max_order+1,N-1):
                k_triples = generate_triples(variables, order=k, all=False, max_number=randomize_higher_order)
                distance_dict[k] = metric_ungraded_DAGs(G_1, G_2, statements_to_be_compared= k_triples, type= type)
                triple_list.extend(k_triples)
                distance += distance_dict[k]

            if normalized == True:
                distance /= (N-1)
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict
            else:
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict

        else:
            if normalized == True:
                distance /= max_order+1
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict
            else:
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict


def metric_directed_cyclic_graphs(graph1, graph2, type ='sc', max_order=None, randomize_higher_order = 0, normalized= True, random_state = None, include_distance_dict = False):
    '''implementation of s/c,s- and c-metric for directed graphs that may contain cycles. Includes random approximations.
           Input:

           graph1, graph2: directed acyclic graphs in the LabelledMixedGraph() format;
           type: options: string, 'sc', 'c' (alternative 'Markov') , 's' (alternative: 'Faithfulness');
           max_order: int, maximal order up to which the metric is computed;
           randomize_higher_order: int, if 0 no randomization, otherwise the number indicates the number of randomly chosen separation statements checked per order > max_order;
           normalized: Boolean,
           random_state: numpy RandomState object for randomization;
           include_distance_dict: Boolean, if True, dictionary of checked statements and result is returned.

        '''

    if graph1.nodes != graph2.nodes:
        raise ValueError('graphs have different nodes!')

    else:

        variables = graph1.nodes
        N = len(variables)

        G_1 = graph1.get_acyclification()

        G_2 = graph2.get_acyclification()



        if max_order is None:
            max_order = N-2

        distance = 0.0
        distance_dict = {}
        triple_list = []

        for k in range(max_order+1):
            k_triples = generate_triples(variables,order=k,all=True)
            if k_triples != []:
                distance_dict[k] = metric_ungraded_DAGs(G_1, G_2, statements_to_be_compared= k_triples, type= type)
                triple_list.extend(k_triples)
                distance += distance_dict[k]


        if randomize_higher_order > 0:

            if random_state == None:
                random_state = np.random

            for k in range(max_order+1,N-1):
                k_triples = generate_triples(variables, order=k, all=False, max_number=randomize_higher_order)
                distance_dict[k] = metric_ungraded_DAGs(G_1, G_2, statements_to_be_compared= k_triples, type= type)
                triple_list.extend(k_triples)
                distance += distance_dict[k]

            if normalized == True:
                distance /= (N-1)
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict
            else:
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict

        else:
            if normalized == True:
                distance /= max_order+1
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict
            else:
                if include_distance_dict == False:
                    return distance
                else:
                    return distance, distance_dict