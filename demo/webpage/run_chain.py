
from src.causalchange.causal_change import CausalChange
from src.causalchange.cc_types import ScoreType, GraphSearch, DataMode
from src.causalchange.gen.generate import gen_example_continuous, to_json_G, gen_example_context

seed = 42
n_nodes = 5

X, truths = gen_example_context(n_nodes=n_nodes, seed=seed)

cc = CausalChange(score_type=ScoreType.SPLINE, graph_search=GraphSearch.CHAIN, data_mode=DataMode.CONTEXTS, vb=4, truths=truths)

G_hat = cc.fit(X)
print("Estimated", G_hat.edges)
print( "True", truths['true_g'].edges)
to_json_G(G_hat, truths, cc, 'run_chain')
