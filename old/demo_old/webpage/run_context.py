
from causalchange.old.causal_change_large import CausalChange
from causalchange._cc_types import ScoreType, GraphSearch, DataMode
from causalchange.gen.generate import to_json_G, gen_example_context

seed = 42
n_nodes = 5

X, truths = gen_example_context(n_nodes=n_nodes, seed=seed)

cc = CausalChange(score_type=ScoreType.SPLINE, graph_search=GraphSearch.GLOBE, data_mode=DataMode.CONTEXTS, vb=4)

G_hat = cc.fit(X)
print("Estimated", G_hat.edges)
print( "True", truths['true_g'].edges)
to_json_G(G_hat, truths, cc, 'run_context')
