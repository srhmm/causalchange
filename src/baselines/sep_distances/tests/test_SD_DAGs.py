from codebase import mixed_graph as mixed
from codebase import metrics

import Graphs_for_testing as G_testing

test_graphs = G_testing.generate_graphs()


G1 = test_graphs['simple_collider']

G2 = test_graphs['chain']

G3 = test_graphs['empty']


#------------------parent-------------------------

print('SD_DAGs(empty,G1,type=parent)')
print(metrics.SD_DAGs(G3,G1,type='parent'))

print('SD_DAGs(empty,G2,type=parent)')
print(metrics.SD_DAGs(G3,G2,type='parent'))

print('SD_DAGs(G1,G2,type=parent)')
print(metrics.SD_DAGs(G1,G2,type='parent'))

print('SD_DAGs(G2,G1,type=parent)')
print(metrics.SD_DAGs(G2,G1,type='parent'))

print('SD_DAGs(G1,G1,type=parent)')
print(metrics.SD_DAGs(G1,G1,type='parent'))

#------------------MB-enhanced parent-------------------------

print('SD_DAGs(empty,G1,type=parent,MB_enhanced=True)')
print(metrics.SD_DAGs(G3,G1,type='parent',MB_enhanced=True))

print('SD_DAGs(empty,G2,type=parent,MB_enhanced=True)')
print(metrics.SD_DAGs(G3,G2,type='parent',MB_enhanced=True))

print('SD_DAGs(G1,G2,type=parent,MB_enhanced=True)')
print(metrics.SD_DAGs(G1,G2,type='parent',MB_enhanced=True))

print('SD_DAGs(G2,G1,type=parent,MB_enhanced=True)')
print(metrics.SD_DAGs(G2,G1,type='parent',MB_enhanced=True))

print('SD_DAGs(G1,G1,type=parent,MB_enhanced=True)')
print(metrics.SD_DAGs(G1,G1,type='parent',MB_enhanced=True))

#------------------ancestor-------------------------

print('SD_DAGs(empty,G1,type=ancestor)')
print(metrics.SD_DAGs(G3,G1,type='ancestor'))

print('SD_DAGs(empty,G2,type=ancestor)')
print(metrics.SD_DAGs(G3,G2,type='ancestor'))

print('SD_DAGs(G1,G2,type=ancestor)')
print(metrics.SD_DAGs(G1,G2,type='ancestor'))

print('SD_DAGs(G2,G1,type=ancestor)')
print(metrics.SD_DAGs(G2,G1,type='ancestor'))

print('SD_DAGs(G1,G1,type=ancestor)')
print(metrics.SD_DAGs(G1,G1,type='ancestor'))

#------------------ZL-------------------------

print('SD_DAGs(empty,G1,type=ZL)')
print(metrics.SD_DAGs(G3,G1,type='ZL'))

print('SD_DAGs(empty,G2,type=ZL)')
print(metrics.SD_DAGs(G3,G2,type='ZL'))

print('SD_DAGs(G1,G2,type=ZL)')
print(metrics.SD_DAGs(G1,G2,type='ZL'))

print('SD_DAGs(G2,G1,type=ZL)')
print(metrics.SD_DAGs(G2,G1,type='ZL'))

print('SD_DAGs(G1,G1,type=ZL)')
print(metrics.SD_DAGs(G1,G1,type='ZL'))

#------------------MB-enhanced ZL-------------------------

print('SD_DAGs(empty,G1,type=ZL,MB_enhanced=True)')
print(metrics.SD_DAGs(G3,G1,type='ZL',MB_enhanced=True))

print('SD_DAGs(empty,G2,type=ZL,MB_enhanced=True)')
print(metrics.SD_DAGs(G3,G2,type='ZL',MB_enhanced=True))

print('SD_DAGs(G1,G2,type=ZL,MB_enhanced=True)')
print(metrics.SD_DAGs(G1,G2,type='ZL',MB_enhanced=True))

print('SD_DAGs(G2,G1,type=ZL,MB_enhanced=True)')
print(metrics.SD_DAGs(G2,G1,type='ZL',MB_enhanced=True))

print('SD_DAGs(G1,G1,type=ZL,MB_enhanced=True)')
print(metrics.SD_DAGs(G1,G1,type='ZL',MB_enhanced=True))