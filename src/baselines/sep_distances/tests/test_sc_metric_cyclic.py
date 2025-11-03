from codebase import mixed_graph as mixed
from codebase import metrics

import Graphs_for_testing as G_testing

test_graphs = G_testing.generate_graphs()


G1 = test_graphs['cyclic']

G2 = test_graphs['chain']

G3 = test_graphs['empty']


print('metric_metric_directed_cyclic_graphs(empty,G1,type =sc, max_order=None, randomize_higher_order = 0, normalized= True)')
print(metrics.metric_directed_cyclic_graphs(G3,G1,type ='sc', max_order=None, randomize_higher_order = 0, normalized= True))

print('metric_metric_directed_cyclic_graphs(empty,G1,type =s, max_order=None, randomize_higher_order = 0, normalized= True)')
print(metrics.metric_directed_cyclic_graphs(G3,G1,type ='s', max_order=None, randomize_higher_order = 0, normalized= True))

print('metric_metric_directed_cyclic_graphs(empty,G1,type =c, max_order=None, randomize_higher_order = 0, normalized= True)')
print(metrics.metric_directed_cyclic_graphs(G3,G1,type ='c', max_order=None, randomize_higher_order = 0, normalized= True))

print('metric_metric_directed_cyclic_graphs(G2,G1,type =sc, max_order=None, randomize_higher_order = 0, normalized= True)')
print(metrics.metric_directed_cyclic_graphs(G2,G1,type ='sc', max_order=None, randomize_higher_order = 0, normalized= True))