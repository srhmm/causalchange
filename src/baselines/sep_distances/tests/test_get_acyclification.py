from codebase import mixed_graph as mixed

import Graphs_for_testing as G_testing

print('We first test a graph that is already a DAG')

test_graphs = G_testing.generate_graphs()

G1 = test_graphs['simple_collider']

print('Graph:')
print('nodes:', G1.nodes)
print(G1)

print('Acyclification:')
print('nodes:', G1.nodes)
print(G1.get_acyclification())

G2 = test_graphs['with_bidirected']

print('Graph with bidirected edges:')
print('nodes:', G2.nodes)
print(G2)

G2_dir = G2.get_canonical_directed_graph()
print('canonical directed:')
print('nodes:', G2_dir.nodes)
print(G2_dir)

print('Acyclification:')
print('nodes:', G2_dir)
print(G2_dir.get_acyclification())




G3 = test_graphs['cyclic']

print('cyclic graph:')
print('nodes:', G3.nodes)
print(G3)

G3_acyclic = G3.get_acyclification()

print('Acyclification:')
print('nodes:', G3_acyclic.nodes)
print(G3_acyclic)
