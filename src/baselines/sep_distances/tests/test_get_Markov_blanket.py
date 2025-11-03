from codebase import mixed_graph as mixed

import Graphs_for_testing as G_testing

test_graphs = G_testing.generate_graphs()


G1 = test_graphs['simple_collider']

G2 = test_graphs['chain']

G1.get_Markov_blankets()
G2.get_Markov_blankets()


print('G1:')
print(G1)
print('Markov blankets:')
print(G1.Markov_blankets)

print('G2:')
print(G2)
print('Markov blankets:')
print(G2.Markov_blankets)