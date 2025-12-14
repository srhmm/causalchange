
from src.causalchange.causal_change import CausalChange
from src.causalchange.cc_types import ScoreType, GraphSearch
from src.causalchange.gen.generate import gen_example_continuous, to_json_G

seed = 42
n_nodes = 5

X, truths = gen_example_continuous(n_nodes=n_nodes, seed=seed)

cc = CausalChange(score_type=ScoreType.GAM, graph_search=GraphSearch.TOPIC, vb=2)

G_hat = cc.fit(X)
print("Estimated", G_hat.edges)
print("True", truths['true_g'].edges)
to_json_G(G_hat, truths, cc, 'run_continuous')
