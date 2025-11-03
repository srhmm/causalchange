from codebase import mixed_graph as mixed
from codebase import metrics

import Graphs_for_testing as G_testing

test_graphs = G_testing.generate_graphs()


G1 = test_graphs['simple_collider']
G2 = test_graphs['chain']

G3 = test_graphs['empty']
G4 = test_graphs['with_bidirected']


#------------------ZL-------------------------

print('SD_mixed_graphs(empty,G1,type=ZL)')
print(metrics.SD_mixed_graphs(G3,G1,type='ZL'))

print('SD_mixed_graphs(empty,G2,type=ZL)')
print(metrics.SD_mixed_graphs(G3,G2,type='ZL'))

print('SD_mixed_graphs(G1,G2,type=ZL)')
print(metrics.SD_mixed_graphs(G1,G2,type='ZL'))

print('SD_mixed_graphs(G2,G1,type=ZL)')
print(metrics.SD_mixed_graphs(G2,G1,type='ZL'))

print('SD_mixed_graphs(G1,G1,type=ZL)')
print(metrics.SD_mixed_graphs(G1,G1,type='ZL'))


print('SD_mixed_graphs(G1,G4,type=ZL)')
print(metrics.SD_mixed_graphs(G1,G4,type='ZL'))


print('SD_mixed_graphs(G4,G1,type=ZL)')
print(metrics.SD_mixed_graphs(G4,G1,type='ZL'))

print('SD_mixed_graphs(empty,G4,type=ZL)')
print(metrics.SD_mixed_graphs(G3,G4,type='ZL'))

print('SD_mixed_graphs(G4,empty,type=ZL)')
print(metrics.SD_mixed_graphs(G4,G3,type='ZL'))

G5 = test_graphs['empty_4_nodes']
G6 = test_graphs['full_4_nodes']

print('SD_mixed_graphs(G5,G6,type=ZL)')
print(metrics.SD_mixed_graphs(G5,G6,type='ZL'))

print('SD_mixed_graphs(G6,G5,type=ZL)')
print(metrics.SD_mixed_graphs(G6,G5,type='ZL'))