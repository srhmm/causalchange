from codebase import mixed_graph as mixed
from codebase import metrics

import Graphs_for_testing as G_testing

test_graphs = G_testing.generate_graphs()


G1 = test_graphs['simple_collider']

G2 = test_graphs['chain']

G3 = test_graphs['empty']

#------------------------------parent-------------------------

print('parent_AID(empty,G1):')
print(metrics.parent_AID_DAGs(G3,G1))

print('parent_AID(empty,G2):')
print(metrics.parent_AID_DAGs(G3,G2))

print('parent_AID(G1,empty):')
print(metrics.parent_AID_DAGs(G1,G3))

print('parent_AID(G2,empty):')
print(metrics.parent_AID_DAGs(G2,G3))

print('parent_AID(G1,G2):')
print(metrics.parent_AID_DAGs(G1,G2,normalized=False))

print('parent_AID(empty,G1):')
print(metrics.sym_parent_AID_DAGs(G3,G1))

print('parent_AID(empty,G2):')
print(metrics.sym_parent_AID_DAGs(G3,G2))

print('parent_AID(G1,G2):')
print(metrics.sym_parent_AID_DAGs(G1,G2,normalized=False))

#------------------ancestor-------------------------

print('ancestor_AID(empty,G1):')
print(metrics.ancestor_AID_DAGs(G3,G1))

print('ancestor_AID(empty,G2):')
print(metrics.ancestor_AID_DAGs(G3,G2))

print('ancestor_AID(G1,empty):')
print(metrics.ancestor_AID_DAGs(G1,G3))

print('ancestor_AID(G2,empty):')
print(metrics.ancestor_AID_DAGs(G2,G3))

print('ancestor_AID(G1,G2):')
print(metrics.ancestor_AID_DAGs(G1,G2,normalized=False))

print('ancestor_AID(empty,G1):')
print(metrics.sym_ancestor_AID_DAGs(G3,G1))

print('ancestor_AID(empty,G2):')
print(metrics.sym_ancestor_AID_DAGs(G3,G2))

print('ancestor_AID(G1,G2):')
print(metrics.sym_ancestor_AID_DAGs(G1,G2,normalized=False))

#---------------------CPDAGS----------------------------

print('--------------CPDAGS----------------')

#------------------------------parent-------------------------

print('parent_AID(empty,G1):')
print(metrics.parent_AID_CPDAGs(G3,G1))

print('parent_AID(empty,G2):')
print(metrics.parent_AID_CPDAGs(G3,G2))

print('parent_AID(G1,empty):')
print(metrics.parent_AID_CPDAGs(G1,G3))

print('parent_AID(G2,empty):')
print(metrics.parent_AID_CPDAGs(G2,G3))

print('parent_AID(G1,G2):')
print(metrics.parent_AID_CPDAGs(G1,G2,normalized=False))

print('parent_AID(empty,G1):')
print(metrics.sym_parent_AID_CPDAGs(G3,G1))

print('parent_AID(empty,G2):')
print(metrics.sym_parent_AID_CPDAGs(G3,G2))

print('parent_AID(G1,G2):')
print(metrics.sym_parent_AID_CPDAGs(G1,G2,normalized=False))

#------------------ancestor-------------------------

print('ancestor_AID(empty,G1):')
print(metrics.ancestor_AID_CPDAGs(G3,G1))

print('ancestor_AID(empty,G2):')
print(metrics.ancestor_AID_CPDAGs(G3,G2))

print('ancestor_AID(G1,empty):')
print(metrics.ancestor_AID_CPDAGs(G1,G3))

print('ancestor_AID(G2,empty):')
print(metrics.ancestor_AID_CPDAGs(G2,G3))

print('ancestor_AID(G1,G2):')
print(metrics.ancestor_AID_CPDAGs(G1,G2,normalized=False))

print('ancestor_AID(empty,G1):')
print(metrics.sym_ancestor_AID_CPDAGs(G3,G1))

print('ancestor_AID(empty,G2):')
print(metrics.sym_ancestor_AID_CPDAGs(G3,G2))

print('ancestor_AID(G1,G2):')
print(metrics.sym_ancestor_AID_CPDAGs(G1,G2,normalized=False))