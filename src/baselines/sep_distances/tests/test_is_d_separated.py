from codebase import mixed_graph as mixed

import Graphs_for_testing as G_testing


test_graphs = G_testing.generate_graphs()

G1 = test_graphs['chain']
G2 = test_graphs['simple_collider']

print('G1')
print(G1)
print('check {X1} {X5} ')
print(G1.is_d_separated(x={'X1',},y={'X5',},z=set(),DAG_check=False))
print('check {X1} {X5} , {X4}')
print(G1.is_d_separated(x={'X1',},y={'X5',},z={'X4'},DAG_check=True))
print('check {X1} {X5} , {X4,X3}')
print(G1.is_d_separated(x={'X1',},y={'X5',},z={'X4','X3'},DAG_check=True))

print('G2')
print(G2)
print('check {X1} {X5} ')
print(G2.is_d_separated(x={'X1',},y={'X5',},z=set(),DAG_check=False))
print('check {X1} {X5} , {X4}')
print(G2.is_d_separated(x={'X1',},y={'X5',},z={'X4'},DAG_check=True))
print('check {X1} {X5} , {X4,X3}')
print(G2.is_d_separated(x={'X1',},y={'X5',},z={'X4','X3'},DAG_check=True))