import numpy as np
import pandas as pd

from src.causalchange.causal_change import CausalChange

X =  pd.DataFrame(np.zeros((30, 4), dtype=float))
g = CausalChange().fit(X)
print(g.edges)