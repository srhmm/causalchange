from codebase import mixed_graph as mixed

import Graphs_for_testing as G_testing


test_graphs = G_testing.generate_graphs()

G1 = test_graphs['chain'].get_CPDAG()


print('nodes:', G1.nodes)
print ('graph:')
print(G1)


print('node: X4')
print('possible ancestors:', G1.get_possible_ancestors('X4'))

print('node: X2')
print('possible ancestors:', G1.get_possible_ancestors('X2'))

G2 = test_graphs['simple_collider'].get_CPDAG()


print('nodes:', G2.nodes)
print ('graph:')
print(G2)


print('node: X4')
print('possible ancestors:', G2.get_possible_ancestors('X4'))

print('node: X2')
print('possible ancestors:', G2.get_possible_ancestors('X2'))

print('node: X5')
print('possible ancestors:', G2.get_possible_ancestors('X5'))