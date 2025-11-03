from codebase import mixed_graph as mixed
from codebase import metrics

import Graphs_for_testing as G_testing

test_graphs = G_testing.generate_graphs()


G1 = test_graphs['simple_collider']

G2 = test_graphs['chain']

G3 = test_graphs['empty']


print('metric_DAGs(empty,G1,type =sc, max_order=None, randomize_higher_order = 0, normalized= True)')
print(metrics.metric_DAGs(G3,G1,type ='sc', max_order=None, randomize_higher_order = 0, normalized= True))

print('metric_DAGs(G1,G1,type =sc, max_order=None, randomize_higher_order = 0, normalized= True)')
print(metrics.metric_DAGs(G1,G1,type ='sc', max_order=None, randomize_higher_order = 0, normalized= True))


print('metric_DAGs(empty,G1,type =sc, max_order=2, randomize_higher_order = 0, normalized= True)')
print(metrics.metric_DAGs(G3,G1,type ='sc', max_order=2, randomize_higher_order = 0, normalized= True))

print('metric_DAGs(G1,G1,type =sc, max_order=2, randomize_higher_order = 0, normalized= True)')
print(metrics.metric_DAGs(G1,G1,type ='sc', max_order=2, randomize_higher_order = 0, normalized= True))

print('metric_DAGs(empty,G1,type =sc, max_order=2, randomize_higher_order = 100, normalized= True)')
print(metrics.metric_DAGs(G3,G1,type ='sc', max_order=2, randomize_higher_order = 100, normalized= True))

print('metric_DAGs(G1,G1,type =sc, max_order=2, randomize_higher_order = 100, normalized= True)')
print(metrics.metric_DAGs(G1,G1,type ='sc', max_order=2, randomize_higher_order = 100, normalized= True))

print('metric_DAGs(empty,G1,type =sc, max_order=2, randomize_higher_order = 100, normalized= False)')
print(metrics.metric_DAGs(G3,G1,type ='sc', max_order=2, randomize_higher_order = 100, normalized= False))

print('metric_DAGs(G1,G1,type =sc, max_order=2, randomize_higher_order = 100, normalized= False)')
print(metrics.metric_DAGs(G1,G1,type ='sc', max_order=2, randomize_higher_order = 100, normalized= False))


print('metric_DAGs(empty,G1,type =sc, max_order=None, randomize_higher_order = 0, normalized= True)')
print(metrics.metric_DAGs(G2,G1,type ='sc', max_order=None, randomize_higher_order = 0, normalized= True))

print('metric_DAGs(G1,G1,type =sc, max_order=None, randomize_higher_order = 0, normalized= True)')
print(metrics.metric_DAGs(G1,G2,type ='sc', max_order=None, randomize_higher_order = 0, normalized= True))
