from codebase import metrics

import Graphs_for_testing as G_testing

test_graphs = G_testing.generate_graphs()

G1 = test_graphs['chain']

G2 = test_graphs['empty']

print('SHD_DAGs(G1,G2,normalized=False)')
print(metrics.SHD_DAGs(G1,G2,normalized=False))

print('SHD_DAGs(G2,G1,normalized=False)')
print(metrics.SHD_DAGs(G2,G1,normalized=False))

print('SHD_DAGs(G1,G2,normalized=True)')
print(metrics.SHD_DAGs(G1,G2,normalized=True))

print('SHD_DAGs(G2,G1,normalized=False)')
print(metrics.SHD_DAGs(G2,G1,normalized=True))

G3 = test_graphs['undirected_chain']

print('SHD_CPDAGs(G1,G2,normalized=False)')
print(metrics.SHD_CPDAGs(G1,G2,normalized=False))

print('SHD_CPDAGs(G2,G1,normalized=False)')
print(metrics.SHD_CPDAGs(G2,G1,normalized=False))

print('SHD_CPDAGs(G1,G2,normalized=True)')
print(metrics.SHD_CPDAGs(G1,G2,normalized=True))

print('SHD_CPDAGs(G2,G1,normalized=False)')
print(metrics.SHD_CPDAGs(G2,G1,normalized=True))

print('SHD_CPDAGs(G3,G2,normalized=True)')
print(metrics.SHD_CPDAGs(G3,G2,normalized=False))

print('SHD_CPDAGs(G2,G3,normalized=False)')
print(metrics.SHD_CPDAGs(G2,G3,normalized=False))

print('SHD_CPDAGs(G1,G3,normalized=False)')
print(metrics.SHD_CPDAGs(G1,G3,normalized=False))

print('SHD_CPDAGs(G3,G1,normalized=False)')
print(metrics.SHD_CPDAGs(G3,G1,normalized=False))

G4 = test_graphs['with_bidirected']

print('SHD_MAGs(G2,G4,normalized=False)')
print(metrics.SHD_MAGs(G2,G4,normalized=False))

print('SHD_MAGs(G1,G4,normalized=False)')
print(metrics.SHD_MAGs(G1,G4,normalized=False))

print('SHD_MAGs(G4,G1,normalized=False)')
print(metrics.SHD_MAGs(G4,G1,normalized=False))