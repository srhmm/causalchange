
def string_graph_to_mixed_graph(graph):
    '''Converts a graph in the format of the Tigramite package to the LabelledMixedGraph class'''
    G = mixed.LabelledMixedGraph()
    for i in range(graph.shape[0]):
        G.add_node(i)
    for (i, j) in zip(*np.where(graph != '')):
        if graph[i, j] == '-->':
            G.add_directed(i,j)
        if graph[i, j] == '<->':
            G.add_bidirected(i,j)
        if graph[i, j] == '---':
            G.add_undirected(i,j)
    return G