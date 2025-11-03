from codebase import mixed_graph as mixed

import Graphs_for_testing as G_testing

test_graphs = G_testing.generate_graphs()

G1 = test_graphs['simple_collider']

print('Simple Collider Graph:')
print('nodes:', G1.nodes)
print(G1)
print('CPDAG:')
print(G1.get_CPDAG())


G2 = test_graphs['chain']

print('Simple Chain Graph:')
print('nodes:', G2.nodes)
print(G2)
print('CPDAG:')
print(G2.get_CPDAG())

G3 = test_graphs['empty']

print('Empty Graph:')
print('nodes:', G3.nodes)
print(G3)
print('CPDAG:')
print(G3.get_CPDAG())

G4 = test_graphs['with_bidirected']

print('Graph with bidirected edges:')
print('nodes:', G4.nodes)
print(G4)
print('CPDAG:')
print(G4.get_CPDAG())