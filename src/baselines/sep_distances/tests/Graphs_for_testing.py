#generates graphs that are used in unit tests

from codebase import mixed_graph as mixed

def generate_graphs():
    nodes = ['X1','X2','X3','X4','X5','X6']

    G1 = mixed.LabelledMixedGraph()
    G2 = mixed.LabelledMixedGraph()
    G3 = mixed.LabelledMixedGraph()
    G4 = mixed.LabelledMixedGraph()
    G5 = mixed.LabelledMixedGraph()
    G6 = mixed.LabelledMixedGraph()

    empty = mixed.LabelledMixedGraph()

    for i in nodes:
        G1.add_node(i)
        G2.add_node(i)
        G3.add_node(i)
        G4.add_node(i)
        G5.add_node(i)
        G6.add_node(i)
        empty.add_node(i)

    #simple causal chain
    G1.add_directed('X1', 'X2')
    G1.add_directed('X2', 'X3')
    G1.add_directed('X3', 'X4')
    G1.add_directed('X4', 'X5')
    G1.add_directed('X5', 'X6')

    #graph with bidirected edges
    G2.add_directed('X1', 'X2')
    G2.add_directed('X2', 'X3')
    G2.add_directed('X3', 'X4')
    G2.add_bidirected('X4', 'X5')
    G2.add_bidirected('X5', 'X6')
    G2.add_bidirected('X1', 'X4')

    G3.add_directed('X1', 'X2')
    G3.add_directed('X2', 'X3')
    G3.add_directed('X3', 'X4')
    G3.add_directed('X5', 'X4')
    G3.add_directed('X5', 'X6')

    # simple undirected chain
    G4.add_undirected('X1', 'X2')
    G4.add_undirected('X2', 'X3')
    G4.add_undirected('X3', 'X4')
    G4.add_undirected('X4', 'X5')
    G4.add_undirected('X5', 'X6')

    # cyclic graph
    G5.add_directed('X1', 'X2')
    G5.add_directed('X2', 'X3')
    G5.add_directed('X3', 'X1')
    G5.add_directed('X6', 'X5')
    G5.add_directed('X5', 'X6')

    # inverse chain
    G6.add_directed('X2', 'X1')
    G6.add_directed('X3', 'X2')
    G6.add_directed('X4', 'X3')
    G6.add_directed('X5', 'X4')
    G6.add_directed('X6', 'X5')

    #smaller graphs

    G7 = mixed.LabelledMixedGraph()
    G8 = mixed.LabelledMixedGraph()

    nodes2 = ['X1', 'X2', 'X3', 'X4']
    for i in nodes2:
        G7.add_node(i)
        G8.add_node(i)

    G7.add_directed('X1', 'X2')
    G7.add_directed('X1', 'X3')
    G7.add_directed('X1', 'X4')
    G7.add_directed('X2', 'X3')
    G7.add_directed('X2', 'X4')
    G7.add_directed('X3', 'X4')




    return {'chain': G1,'with_bidirected': G2,'simple_collider': G3,'undirected_chain': G4,'cyclic': G5, 'inverse_chain': G6, 'empty': empty, 'full_4_nodes': G7, 'empty_4_nodes': G8 }


