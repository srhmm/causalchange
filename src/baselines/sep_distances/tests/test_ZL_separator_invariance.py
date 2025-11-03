from codebase import mixed_graph as mixed

import Graphs_for_testing as G_testing


test_graphs = G_testing.generate_graphs()

G1 = test_graphs['chain']

G2 = test_graphs['undirected_chain']

G2.get_representative_of_MEC()

G3 = G2._representative

G4 = test_graphs['inverse_chain']

print('G1:')
print(G1)
print('G3:')
print(G3)
print('G4:')
print(G4)

print('Node pair:')
print('X1, X3')
print('ZL-separator on G1:', G1.find_minimal_d_separator({'X1',},{'X3',}))
print('ZL-separator on G3:', G3.find_minimal_d_separator({'X1',},{'X3',}))
print('ZL-separator on G4:', G4.find_minimal_d_separator({'X1',},{'X3',}))


print('Node pair:')
print('X1, X6')
print('ZL-separator on G1:', G1.find_minimal_d_separator({'X1',},{'X6',}))
print('ZL-separator on G3:', G3.find_minimal_d_separator({'X1',},{'X6',}))
print('ZL-separator on G4:', G4.find_minimal_d_separator({'X1',},{'X6',}))

print('Node pair:')
print('X5, X2')
print('ZL-separator on G1:', G1.find_minimal_d_separator({'X5',},{'X2',}))
print('ZL-separator on G3:', G3.find_minimal_d_separator({'X5',},{'X2',}))
print('ZL-separator on G4:', G4.find_minimal_d_separator({'X5',},{'X2',}))

G5 = test_graphs['simple_collider']

G6 = G5.get_CPDAG()
G6.get_representative_of_MEC()

G7 = G6._representative

print('G5:')
print(G5)
print('G7:')
print(G7)

print('Nodes:')
print('{X6,X3}, X1')
print('ZL-separator on G5:', G5.find_minimal_d_separator({'X6','X3'},{'X1',}))
print('ZL-separator on G7:', G7.find_minimal_d_separator({'X6','X3'},{'X1',}))
