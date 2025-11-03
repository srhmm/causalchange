from codebase import mixed_graph as mixed

import Graphs_for_testing as G_testing

test_graphs = G_testing.generate_graphs()

G1 = test_graphs['simple_collider']

print('Simple Collider Graph:')
print('nodes:', G1.nodes)
print(G1)
print('Causal Order:')
print(G1.causal_order())

G2 = test_graphs['with_bidirected']

print('Graph with bidirected edges:')
print('nodes:', G2.nodes)
print(G2)
print('Causal Order:')
print(G2.causal_order())

G3 = test_graphs['cyclic']

print('Graph with cycles:')
print('nodes:', G3.nodes)
print(G3)
print('Causal Order:')
print(G3.causal_order())