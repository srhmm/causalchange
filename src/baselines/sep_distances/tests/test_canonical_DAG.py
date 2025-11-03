from codebase import mixed_graph as mixed

import Graphs_for_testing as G_testing

test_graphs = G_testing.generate_graphs()

G1 = test_graphs['chain']

print('Do the following two graphs coincide?')
print('Graph:')
print('nodes:', G1.nodes)
print(G1)


print('canonical directed:')
print('nodes:', G1.get_canonical_directed_graph().nodes)
print(G1.get_canonical_directed_graph())


#G1.add_bidirected('X4', 'X5')
#G1.add_bidirected('X5', 'X6')
#G1.add_bidirected('X1', 'X4')

G2 = test_graphs['with_bidirected']

print('Graph and canonical directed graph:')

print('Graph:')
print('nodes:', G2.nodes)
print(G2)


print('canonical directed:')
print('nodes:', G2.get_canonical_directed_graph().nodes)
print(G2.get_canonical_directed_graph())
