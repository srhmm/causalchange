from codebase import mixed_graph as mixed

import Graphs_for_testing as G_testing


test_graphs = G_testing.generate_graphs()

G1 = test_graphs['chain']


G2 = G1.get_CPDAG()

print('nodes:', G2.nodes)
print ('graph:')
print(G2)


print('node: X4')
print('pparents:', G2.possibleparents_of('X4'))

print('node: X2')
print('pparents:', G2.possibleparents_of('X2'))

G3 = test_graphs['simple_collider'].get_CPDAG()


print('nodes:', G3.nodes)
print ('graph:')
print(G3)


print('node: X4')
print('pparents:',G3.possibleparents_of('X4'))

print('node: X2')
print('pparents:',G3.possibleparents_of('X2'))

print('node: X3')
print('parents:',G3.parents_of('X3'))
print('pparents:',G3.possibleparents_of('X3'))

print('node: X5')
print('pparents:',G3.possibleparents_of('X5'))

print(G3.directed_keys)