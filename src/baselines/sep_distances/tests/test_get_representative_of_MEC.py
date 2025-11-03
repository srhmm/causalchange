from codebase import mixed_graph as mixed

import Graphs_for_testing as G_testing

test_graphs = G_testing.generate_graphs()


G1 = test_graphs['simple_collider'].get_CPDAG()
G1.get_representative_of_MEC()

print('CPDAG:')
print('nodes:', G1.nodes)
print(G1)
print('Representative:')
print(G1._representative)


G2 = test_graphs['chain'].get_CPDAG()
G2.get_representative_of_MEC()

print('CPDAG:')
print('nodes:', G2.nodes)
print(G2)
print('Representative:')
print(G2._representative)

G3 = test_graphs['empty'].get_CPDAG()
G3.get_representative_of_MEC()

print('CPDAG:')
print('nodes:', G3.nodes)
print(G3)
print('Representative:')
print(G3._representative)
