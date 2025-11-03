from codebase import mixed_graph as mixed
from codebase import metrics

import Graphs_for_testing as G_testing

test_graphs = G_testing.generate_graphs()


G1 = test_graphs['simple_collider'].get_CPDAG()
print(G1)
G2 = test_graphs['undirected_chain']

G3 = test_graphs['empty']


#------------------pparent-------------------------

print('SD_CPDAGs(empty,G1,type=pparent)')
print(metrics.SD_CPDAGs(G3,G1,type='pparent'))

print('SD_CPDAGs(empty,G2,type=pparent)')
print(metrics.SD_CPDAGs(G3,G2,type='pparent'))

print('SD_CPDAGs(G1,G2,type=pparent)')
print(metrics.SD_CPDAGs(G1,G2,type='pparent'))

print('SD_CPDAGs(G2,G1,type=pparent)')
print(metrics.SD_CPDAGs(G2,G1,type='pparent'))

print('SD_CPDAGs(G1,G1,type=pparent)')
print(metrics.SD_CPDAGs(G1,G1,type='pparent'))

#------------------MB-enhanced pparent-------------------------

print('SD_CPDAGs(empty,G1,type=pparent,MB_enhanced=True)')
print(metrics.SD_CPDAGs(G3,G1,type='pparent',MB_enhanced=True))

print('SD_CPDAGs(empty,G2,type=pparent,MB_enhanced=True)')
print(metrics.SD_CPDAGs(G3,G2,type='pparent',MB_enhanced=True))

print('SD_CPDAGs(G1,G2,type=pparent,MB_enhanced=True)')
print(metrics.SD_CPDAGs(G1,G2,type='pparent',MB_enhanced=True))

print('SD_CPDAGs(G2,G1,type=pparent,MB_enhanced=True)')
print(metrics.SD_CPDAGs(G2,G1,type='pparent',MB_enhanced=True))

print('SD_CPDAGs(G1,G1,type=pparent,MB_enhanced=True)')
print(metrics.SD_CPDAGs(G1,G1,type='pparent',MB_enhanced=True))

#------------------pancestor-------------------------

print('SD_CPDAGs(empty,G1,type=pancestor)')
print(metrics.SD_CPDAGs(G3,G1,type='pancestor'))

print('SD_CPDAGs(empty,G2,type=pancestor)')
print(metrics.SD_CPDAGs(G3,G2,type='pancestor'))

print('SD_CPDAGs(G1,G2,type=pancestor)')
print(metrics.SD_CPDAGs(G1,G2,type='pancestor'))

print('SD_CPDAGs(G2,G1,type=pancestor)')
print(metrics.SD_CPDAGs(G2,G1,type='pancestor'))

print('SD_CPDAGs(G1,G1,type=pancestor)')
print(metrics.SD_CPDAGs(G1,G1,type='pancestor'))

#------------------ZL-------------------------

print('SD_CPDAGs(empty,G1,type=ZL)')
print(metrics.SD_CPDAGs(G3,G1,type='ZL'))

print('SD_CPDAGs(empty,G2,type=ZL)')
print(metrics.SD_CPDAGs(G3,G2,type='ZL'))

print('SD_CPDAGs(G1,G2,type=ZL)')
print(metrics.SD_CPDAGs(G1,G2,type='ZL'))

print('SD_CPDAGs(G2,G1,type=ZL)')
print(metrics.SD_CPDAGs(G2,G1,type='ZL'))

print('SD_CPDAGs(G1,G1,type=ZL)')
print(metrics.SD_CPDAGs(G1,G1,type='ZL'))

#------------------MB-enhanced ZL-------------------------

print('SD_CPDAGs(empty,G1,type=ZL,MB_enhanced=True)')
print(metrics.SD_CPDAGs(G3,G1,type='ZL',MB_enhanced=True))

print('SD_CPDAGs(empty,G2,type=ZL,MB_enhanced=True)')
print(metrics.SD_CPDAGs(G3,G2,type='ZL',MB_enhanced=True))

print('SD_CPDAGs(G1,G2,type=ZL,MB_enhanced=True)')
print(metrics.SD_CPDAGs(G1,G2,type='ZL',MB_enhanced=True))

print('SD_CPDAGs(G2,G1,type=ZL,MB_enhanced=True)')
print(metrics.SD_CPDAGs(G2,G1,type='ZL',MB_enhanced=True))

print('SD_CPDAGs(G1,G1,type=ZL,MB_enhanced=True)')
print(metrics.SD_CPDAGs(G1,G1,type='ZL',MB_enhanced=True))