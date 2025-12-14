import numpy as np
import networkx as nx
import pytest

from causalchange.scoring.fit_cond_mixture import MixingType
from src.causalchange.causal_change import CausalChange
from src.causalchange.cc_types import DataMode, GraphSearch, GPType


def _assert_sane_graph(G, N):
    assert isinstance(G, nx.DiGraph)
    assert set(G.nodes) == set(range(N))
    assert nx.is_directed_acyclic_graph(G)
    assert all(u != v for u, v in G.edges)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "data_mode",
    [DataMode.IID, DataMode.CONTEXTS, DataMode.MIXED, DataMode.TIME, DataMode.TIME_CONTEXTS],
)
@pytest.mark.parametrize(
    "graph_search",
    [GraphSearch.TOPIC, GraphSearch.GLOBE, GraphSearch.CHAIN],
)
def test_fit_smoke_across_compatible_modes(data_mode, graph_search):
    if not graph_search.is_compatible_with(data_mode):
        pytest.skip(f"{graph_search} incompatible with {data_mode}")

    N = 4
    rng = np.random.default_rng(123)

    # Provide minimal data for each mode
    if data_mode == DataMode.IID or data_mode == DataMode.MIXED:
        X = rng.standard_normal((80, N))
    elif data_mode == DataMode.CONTEXTS or data_mode == DataMode.TIME_CONTEXTS:
        X = {0: rng.standard_normal((60, N)), 1: rng.standard_normal((70, N))}
    elif data_mode == DataMode.TIME:
        X = rng.standard_normal((80, N))
    else:
        pytest.skip("Unhandled DataMode")

    cc = CausalChange(
        data_mode=data_mode,
        graph_search=graph_search,
        score_type=GPType.EXACT,
        mixing_type=MixingType.MIX_LIN if data_mode == DataMode.MIXED else MixingType.SKIP,
        vb=0,
    )

    try:
        G = cc.fit(X)
    except NotImplementedError:
        pytest.xfail(f"{data_mode} not implemented yet")

    _assert_sane_graph(G, N)

    # Optional: only assert topo order exists for modes that define it
    if hasattr(cc, "topological_order") and cc.topological_order is not None:
        topo = list(cc.topological_order)
        assert sorted(topo) == sorted(G.nodes)
