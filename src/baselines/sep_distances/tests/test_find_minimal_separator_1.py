from codebase import mixed_graph as mixed

import Graphs_for_testing as G_testing


test_graphs = G_testing.generate_graphs()

G1 = test_graphs['chain']



print('nodes:', G1.nodes)
print ('graph:')
print(G1)

print('Node pair:')
print('X1, X2')
print('ZL-separator:', G1.find_minimal_d_separator({'X1',},{'X2',}))

print('Node pair:')
print('X2, X1')
print('ZL-separator:', G1.find_minimal_d_separator({'X2',},{'X1',}))

print('Node pair:')
print('X1, X3')
print('ZL-separator:', G1.find_minimal_d_separator({'X1',},{'X3',}))

print('Node pair:')
print('X3, X1')
print('ZL-separator:', G1.find_minimal_d_separator({'X3',},{'X1',}))

print('Node pair:')
print('X1, X4')
print('ZL-separator:', G1.find_minimal_d_separator({'X1',},{'X4',}))

print('alternative:')

import networkx as nx

print(nx.find_minimal_d_separator(G1.nx_graph,{'X1',},{'X4',}))
