
''' this file is an extension of the LabellelMixedGraph class implemented by Chandler Squires https://github.com/csquires/dct-policy'''
import copy
import itertools
from itertools import chain
from collections import defaultdict
import networkx as nx
from typing import Union
import collections
from collections import deque


class LabelledMixedGraph:
    def __init__(self, nodes=set(), directed=dict(), undirected=dict(), bidirected=dict(), semidirected=dict()):
        self._nodes = set(nodes)
        self._directed = {(i, j): label for (i, j), label in directed.items()}
        self._bidirected = {frozenset({i, j}): label for (i, j), label in bidirected.items()}
        self._undirected = {frozenset({i, j}): label for (i, j), label in undirected.items()}
        '''Undirected edges are interpreted as representation of edges that can't be oriented in a Markov equivalence class. 
        They do not indicate selection bias.'''

        self._semidirected = {(i, j): label for (i, j), label in semidirected.items()}

        self._neighbors = defaultdict(set)
        self._spouses = defaultdict(set)
        self._parents = defaultdict(set)
        self._children = defaultdict(set)
        self._semiparents = defaultdict(set)
        self._semichildren = defaultdict(set)
        self._possibleparents = defaultdict(set)
        self._ancestors = defaultdict(set)

        self._representative = None

        self.nx_graph = None

        self.DAG = None

        self.Markov_blankets = None

        for i, j in self._directed.keys():
            self._children[i].add(j)
            self._parents[j].add(i)
            self._possibleparents[j].add(i)
            self._nodes.add(i)
            self._nodes.add(j)

        for i, j in self._bidirected.keys():
            self._spouses[j].add(i)
            self._spouses[i].add(j)
            self._nodes.add(i)
            self._nodes.add(j)
        for i, j in self._undirected.keys():
            self._neighbors[i].add(j)
            self._neighbors[j].add(i)
            self._possibleparents[j].add(i)
            self._nodes.add(i)
            self._nodes.add(j)

        for i, j in self._semidirected.keys():
            self._semichildren[i].add(j)
            self._semiparents[j].add(i)
            self._nodes.add(i)
            self._nodes.add(j)

    # === BUILT-INS
    def __str__(self):
        s = ""
        s += f"Directed edges: {self._directed}\n"
        s += f"Undirected edges: {self._undirected}\n"
        s += f"Bidirected edges: {self._bidirected}\n"
        s += f"Semidirected edges: {self._semidirected}\n"
        return s

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        same_dir = self._directed == other._directed
        same_bidir = self._bidirected == other._bidirected
        same_undir = self._undirected == other._undirected
        same_semidir = self._semidirected == other._semidirected
        return same_dir and same_bidir and same_undir and same_semidir

    def copy(self):
        return LabelledMixedGraph(
            nodes=self._nodes,
            directed=self._directed,
            bidirected=self._bidirected,
            undirected=self._undirected,
            semidirected=self._semidirected
        )

    # === PROPERTIES
    @property
    def nodes(self):
        return set(self._nodes)

    @property
    def node_list(self):
        return list(self._nodes)

    @property
    def directed(self):
        return dict(self._directed)

    @property
    def undirected(self):
        return dict(self._undirected)

    @property
    def bidirected(self):
        return dict(self._bidirected)

    @property
    def semidirected(self):
        return dict(self._semidirected)

    @property
    def directed_keys(self):
        return set(self._directed.keys())

    @property
    def bidirected_keys(self):
        return set(self._bidirected.keys())

    @property
    def undirected_keys(self):
        return set(self._undirected.keys())

    @property
    def semidirected_keys(self):
        return set(self._semidirected.keys())

    @property
    def nnodes(self):
        return len(self._nodes)

    @property
    def num_directed(self):
        return len(self._directed)

    @property
    def num_undirected(self):
        return len(self._undirected)

    @property
    def num_bidirected(self):
        return len(self._bidirected)

    @property
    def num_semidirected(self):
        return len(self._semidirected)

    @property
    def num_edges(self):
        return self.num_bidirected + self.num_directed + self.num_undirected + self.num_semidirected

    # === CONVERTERS
    @classmethod
    def from_nx(cls, nx_graph):
        """
        Create a LabelledMixedGraph from a networkx graph with labelled edges.
        """
        if isinstance(nx_graph, nx.MultiDiGraph) or isinstance(nx_graph, nx.DiGraph):
            directed = {(i, j): nx_graph.get_edge_data(i, j) for i, j in nx_graph.edges()}
            bidirected_keys = set(directed.keys()) & {(j, i) for i, j in directed}
            bidirected = {(i, j): directed[(i, j)] for i, j in bidirected_keys}
            directed = {
                (i, j): val for (i, j), val in directed.items()
                if (i, j) not in bidirected_keys and (j, i) not in bidirected_keys
            }
            return LabelledMixedGraph(nodes=nx_graph.nodes(), directed=directed, bidirected=bidirected)
        elif isinstance(nx_graph, nx.Graph):
            undirected = {frozenset({i, j}): nx_graph.get_edge_data(i, j) for i, j in nx_graph.edges()}
            return LabelledMixedGraph(nodes=nx_graph.nodes(), undirected=undirected)

    def to_nx(self) -> Union[nx.Graph, nx.DiGraph]:
        """
        Return a networkx graph. If the current graph has no undirected edges, return a DiGraph.
        If it has no directed or undirected edges, return an undirected Graph.
        """
        if not self._undirected:
            nx_graph = nx.DiGraph()
            nx_graph.add_nodes_from(self._nodes)
            nx_graph.add_edges_from(self._directed.keys())
            bidirected = {(i, j) for i, j in self._bidirected.keys()}
            nx_graph.add_edges_from(bidirected | {(j, i) for i, j in bidirected})
            nx.set_edge_attributes(nx_graph, self._directed, name='label')
            nx.set_edge_attributes(nx_graph, self._bidirected, name='label')
            nx.set_edge_attributes(nx_graph, {(j, i): l for (i, j), l in self.bidirected.items()}, name='label')
            return nx_graph
        if not self._directed and not self._bidirected:
            nx_graph = nx.Graph()
            nx_graph.add_nodes_from(self._nodes)
            nx_graph.add_edges_from(self._undirected.keys())
            nx.set_edge_attributes(nx_graph, self._undirected, 'label')
            return nx_graph
        else:
            raise ValueError("Can only convert if the graph has only undirected edges or no undirected edges")

    def induced_graph(self, nodes):
        """Return the induced subgraph of this graph over `nodes`."""
        return LabelledMixedGraph(
            nodes,
            directed={(i, j): val for (i, j), val in self._directed.items() if i in nodes and j in nodes},
            bidirected={(i, j): val for (i, j), val in self._bidirected.items() if i in nodes and j in nodes},
            undirected={(i, j): val for (i, j), val in self._undirected.items() if i in nodes and j in nodes},
            semidirected={(i, j): val for (i, j), val in self._semidirected.items() if i in nodes and j in nodes}
        )

    def undirected_copy(self):
        """Return a copy of this graph with all edges undirected."""
        edges = {
            **self._undirected,
            **self._bidirected,
            **{frozenset({i, j}): label for (i, j), label in self._directed.items()},
            **{frozenset({i, j}): label for (i, j), label in self._semidirected.items()}
        }
        return LabelledMixedGraph(nodes=self._nodes, undirected=edges)

    # === PREDICATES
    def has_directed(self, i, j):
        """Check if this graph has the directed edge i->j."""
        return (i, j) in self._directed

    def has_bidirected(self, edge):
        """Check if this graph has the bidirected edge `edge`."""
        return frozenset({*edge}) in self._bidirected

    def has_undirected(self, edge):
        """Check if this graph has the undirected edge `edge`."""
        return frozenset({*edge}) in self._undirected

    def has_semidirected(self, i, j):
        """Check if this graph has the semidirected edge i*->j."""
        return (i, j) in self._semidirected

    def has_any_edge(self, edge):
        """Check if this graph has any edge `edge`."""
        i, j = edge
        return self.has_directed(i, j) or self.has_directed(j, i) or self.has_bidirected(edge) \
            or self.has_undirected(edge) or self.has_semidirected(i, j) or self.has_semidirected(j, i)

    # === NODE-WISE SETS
    def indegree_of(self, node):
        """Return the number of parents of a node"""
        return len(self._parents[node])

    def outdegree_of(self, node):
        """Return the number of children of a node"""
        return len(self._children[node])

    def spouse_degree_of(self, node):
        """Return the number of spouses of a node"""
        return len(self._spouses[node])

    def neighbor_degree_of(self, node):
        """Return the number of neighbors of a node"""
        return len(self._neighbors[node])

    def semi_indegree_of(self, node):
        """Return the number of semi-parents of a node"""
        return len(self._parents[node])

    def semi_outdegree_of(self, node):
        """Return the number of semi-children of a node"""
        return len(self._children[node])

    def neighbors_of(self, node):
        return set(self._neighbors[node])

    def spouses_of(self, node):
        return set(self._spouses[node])

    def children_of(self, node):
        return set(self._children[node])

    def parents_of(self, node):
        return set(self._parents[node])

    def semichildren_of(self, node):
        return set(self._semichildren[node])

    def semiparents_of(self, node):
        return set(self._semiparents[node])

    def possibleparents_of(self, node):
        return set(self._possibleparents[node])

    def adjacent_to(self, node):
        """Return all nodes adjacent to `node`."""
        return self.parents_of(node) | self.children_of(node) | self.spouses_of(node) | self.neighbors_of(node) \
            | self.semiparents_of(node) | self.semichildren_of(node)

    def onto_edges(self, node):
        """
        Return all edges with an arrowhead at `node`.
        """
        directed_onto = {(p, node): self._directed[(p, node)] for p in self._parents[node]}
        bidirected_onto = {(s, node): self._bidirected[frozenset({s, node})] for s in self._spouses[node]}
        semi_directed_onto = {(p, node): self._semidirected[(p, node)] for p in self._semiparents[node]}
        return {**directed_onto, **bidirected_onto, **semi_directed_onto}

    def onto_nodes(self, node):
        """Return all parents, spouses, and semiparents of `node`."""
        return {*self._parents[node], *self._spouses[node], *self._semiparents[node]}

    # === EDGE FUNCTIONALS
    def get_label(self, edge, ignore_error=True):
        i, j = edge
        label = self._directed.get((i, j))
        if label: return label
        label = self._directed.get((j, i))
        if label: return label
        label = self._undirected.get(frozenset({*edge}))
        if label: return label
        label = self._bidirected.get(frozenset({*edge}))
        if label: return label
        label = self._semidirected.get((i, j))
        if label: return label
        label = self._semidirected.get((j, i))
        if label: return label

        if not ignore_error:
            raise KeyError(f"No edge {edge}")

    def directed_edges_with_label(self, label):
        return {edge for edge, l in self._directed.items() if l == label}

    def undirected_edges_with_label(self, label):
        return {edge for edge, l in self._undirected.items() if l == label}

    def bidirected_edges_with_label(self, label):
        return {edge for edge, l in self._bidirected.items() if l == label}

    def semidirected_edges_with_label(self, label):
        return {edge for edge, l in self._semidirected.items() if l == label}

    # === ADDERS

    def add_node(self,i):
        self._nodes.add(i)

    def add_directed(self, i, j, label=None):
        """Add i->j with label `label` to this graph."""
        self._directed[(i, j)] = label
        self._parents[j].add(i)
        self._possibleparents[j].add(i)
        self._children[i].add(j)

    def add_bidirected(self, i, j, label=None):
        """Add i<->j with label `label` to this graph."""
        self._bidirected[frozenset({i, j})] = label
        self._spouses[j].add(i)
        self._spouses[i].add(j)

    def add_undirected(self, i, j, label=None):
        """Add i-j with label `label` to this graph."""
        self._undirected[frozenset({i, j})] = label
        self._neighbors[j].add(i)
        self._neighbors[i].add(j)
        self._possibleparents[j].add(i)
        self._possibleparents[i].add(j)

    def add_semidirected(self, i, j, label=None):
        """Add i*->j with label `label` to this graph."""
        self._semidirected[(i, j)] = label
        self._semiparents[j].add(i)
        self._semichildren[i].add(j)

    # === REMOVERS
    def remove_node(self, i):
        """
        Remove the node `i`, and all incident edges, from this graph.
        """
        self._nodes.remove(i)

        for parent in self._parents[i]:
            self._children[parent].remove(i)
        del self._parents[i]

        del self._possibleparents[i]

        for child in self._children[i]:
            self._parents[child].remove(i)
        del self._children[i]

        for spouse in self._spouses[i]:
            self._spouses[spouse].remove(i)
        del self._spouses[i]

        for nbr in self._neighbors[i]:
            self._neighbors[nbr].remove(i)
        del self._neighbors[i]

        for sp in self._semiparents[i]:
            self._semichildren[sp].remove(i)
        del self._semiparents[i]

        for sc in self._semichildren[i]:
            self._semiparents[sc].remove(i)
        del self._semichildren[i]

        self._directed = {(j, k): val for (j, k), val in self._directed.items() if i != j and i != k}
        self._bidirected = {frozenset({j, k}): val for (j, k), val in self._bidirected.items() if i != j and i != k}
        self._undirected = {frozenset({j, k}): val for (j, k), val in self._undirected.items() if i != j and i != k}
        self._semidirected = {(j, k): val for (j, k), val in self._semidirected.items() if i != j and i != k}

    def remove_directed(self, i, j, ignore_error=True):
        """Remove a directed edge i->j from this graph."""
        try:
            label = self._directed.pop((i, j))
            self._parents[j].remove(i)
            self._possibleparents[j].remove(i)
            self._children[i].remove(j)
            return label
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_bidirected(self, i, j, ignore_error=True):
        """Remove a bidirected edge i<->j from this graph."""
        try:
            label = self._bidirected.pop(frozenset({(i, j)}))
            self._spouses[i].remove(j)
            self._spouses[j].remove(i)
            return label
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_undirected(self, i, j, ignore_error=True):
        """Remove an undirected edge i-j from this graph."""
        try:
            label = self._undirected.pop(frozenset({i, j}))
            self._neighbors[i].remove(j)
            self._neighbors[j].remove(i)
            self._possibleparents[i].remove(j)
            self._possibleparents[j].remove(i)
            return label
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_semidirected(self, i, j, ignore_error=True):
        """Remove a semidirected edge i*->j from this graph."""
        try:
            label = self._semidirected.pop((i, j))
            self._semiparents[j].remove(i)
            self._semichildren[i].remove(j)
            return label
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_edge(self, i, j, ignore_error=True):
        """
        Remove the edge between i and j in this graph, if any exists.
        """
        label = self.remove_directed(i, j)
        if label: return label
        label = self.remove_directed(j, i)
        if label: return label
        label = self.remove_bidirected(i, j)
        if label: return label
        label = self.remove_undirected(i, j)
        if label: return label
        label = self.remove_semidirected(i, j)
        if label: return label
        label = self.remove_semidirected(j, i)
        if label: return label

        if not label and not ignore_error:
            raise KeyError("i-j is not an edge in this graph")

    def remove_edges(self, edges, ignore_error=True):
        """
        Remove all edges in `edges` from this graph.
        """
        for edge in edges:
            self.remove_edge(*edge, ignore_error=ignore_error)

    def remove_all_directed(self):
        """Remove all directed edges from this graph."""
        for i, j in self._directed:
            self._parents[j].remove(i)
            self._children[i].remove(j)
        self._directed = defaultdict(set)

    def remove_all_bidirected(self):
        """Remove all bidirected edges from this graph."""
        for i, j in self._bidirected:
            self._spouses[i].remove(j)
            self._spouses[j].remove(i)
        self._bidirected = defaultdict(set)

    def remove_all_undirected(self):
        """Remove all undirected edges from this graph."""
        for i, j in self._undirected:
            self._neighbors[i].remove(j)
            self._neighbors[j].remove(i)
        self._undirected = defaultdict(set)

    def remove_all_semidirected(self):
        """Remove all semidirected edges from this graph."""
        for i, j in self._semidirected:
            self._semiparents[j].remove(i)
            self._semichildren[i].remove(j)
        self._semidirected = defaultdict(set)

    # === MUTATORS
    def to_directed(self, i, j, check_exists=True):
        """Replace the edge between i and j, if any exists, with i->j"""
        label = self.remove_bidirected(i, j)
        label = self.remove_undirected(i, j) if label is None else label
        label = self.remove_semidirected(i, j) if label is None else label
        label = self.remove_semidirected(j, i) if label is None else label
        if label or not check_exists:
            self.add_directed(i, j, label)

    def to_bidirected(self, i, j, check_exists=True):
        """Replace the edge between i and j, if any exists, with i<->j"""
        label = self.remove_undirected(i, j)
        label = self.remove_directed(i, j) if label is None else label
        label = self.remove_directed(j, i) if label is None else label
        label = self.remove_semidirected(i, j) if label is None else label
        label = self.remove_semidirected(j, i) if label is None else label
        if label or not check_exists:
            self.add_bidirected(i, j, label)

    def to_undirected(self, i, j, check_exists=True):
        """Replace the edge between i and j, if any exists, with i-j"""
        label = self.remove_bidirected(i, j)
        label = self.remove_directed(i, j) if label is None else label
        label = self.remove_directed(j, i) if label is None else label
        label = self.remove_semidirected(i, j) if label is None else label
        label = self.remove_semidirected(j, i) if label is None else label
        if label or not check_exists:
            self.add_undirected(i, j, label)

    def to_semidirected(self, i, j, check_exists=True):
        """Replace the edge between i and j, if any exists, with i o-> j"""
        label = self.remove_undirected(i, j)
        label = self.remove_bidirected(i, j) if label is None else label
        label = self.remove_directed(i, j) if label is None else label
        label = self.remove_directed(j, i) if label is None else label
        if label or not check_exists:
            self.add_semidirected(i, j, label)

    def all_to_undirected(self):
        """
        Change all edges in this graph into undirected edges.
        """
        self._undirected.update({frozenset({i, j}): label for (i, j), label in self._directed.items()})
        self._undirected.update({frozenset({i, j}): label for (i, j), label in self._bidirected.items()})
        self._undirected.update({frozenset({i, j}): label for (i, j), label in self._semidirected.items()})
        self.remove_all_directed()
        self.remove_all_bidirected()
        self.remove_all_semidirected()

    def get_canonical_directed_graph(self):
        G = copy.deepcopy(self)
        G.latents = set()
        #G.selectors = set()
        for edge in G._bidirected.keys():
            node_name = 'L_'+str(tuple(edge)[0])+'_'+str(tuple(edge)[1])
            G.add_node(node_name)
            G.add_directed(node_name,tuple(edge)[0])
            G.add_directed(node_name, tuple(edge)[1])
            G.latents.add(node_name)

        #for edge in G._undirected.keys():
        #    node_name = 'S'+str(tuple(edge)[0])+str(tuple(edge)[1])
        #    G.add_node(node_name)
        #    G.add_directed(tuple(edge)[0],node_name)
        #    G.add_directed(tuple(edge)[1],node_name)
        #    G.selectors.add(node_name)

        #G.remove_all_undirected()
        G.remove_all_bidirected()

        return G

    def get_acyclification(self):
        if len(self._bidirected) > 0 or len(self._undirected)>0:
            raise ValueError('acyclification is only implemented for directed graphs. For mixed graphs apply get_canonical_directed_graph() first.')

        G = copy.deepcopy(self).to_nx()

        strongly_connected_components = sorted(nx.strongly_connected_components(G),key=len,reverse=True)

        for c in strongly_connected_components:
            if len(c) > 1:
                G.add_node('SCC'+str(strongly_connected_components.index(c)))
                for node in c:
                    G.add_edge('SCC'+str(strongly_connected_components.index(c)),node)

                scc_pred = set()

                for node in c:
                    scc_pred.union(set(G.predecessors(node)))

                for (parent,node) in itertools.product(scc_pred,c):
                    G.add_edge(parent,node)

                for (node1,node2) in itertools.product(c,c):
                    if G.has_edge(node1,node2):
                        G.remove_edge(node1,node2)
                    if G.has_edge(node2,node1):
                        G.remove_edge(node2,node1)
        G_acyclified = LabelledMixedGraph()

        return G_acyclified.from_nx(G)

    #def get_canonical_acyclification(self):

        #if isinstance(self._bidirected,collections.defaultdict) and isinstance(self._undirected,collections.defaultdict):
        #    return self.get_acyclification()

        #else:
     #   G = self.get_canonical_directed_graph()
    #  return G.get_acyclification()

    def causal_order(self):
        '''Computes a causal order of a mixed graph without cycles'''

        nodes = copy.deepcopy(self.node_list)
        causal_order = []
        length = copy.deepcopy(len(nodes))
        while len(nodes) > 0:
            for node in nodes:
                if self.parents_of(node).isdisjoint(set(nodes).difference({node})):
                    causal_order.append(node)
                    nodes.remove(node)
            if len(nodes) == length:
                raise ValueError('No exogeneous node found, your graph is cyclic')
            else: length = copy.deepcopy(len(nodes))

        return causal_order

    def get_representative_of_MEC(self):
        ''' returns a representative of a Markov equivalence class of DAGs as represented by a CPDAG'''

        def rule1(skeleton):
            action_taken = False
            to_be_oriented = []
            for (i, j) in skeleton._directed.keys():
                for k in skeleton.neighbors_of(j):
                    if i not in skeleton.adjacent_to(k):
                        to_be_oriented.append((j,k))
                        action_taken = True
            for (j,k) in to_be_oriented:
                skeleton.add_directed(j, k)
                skeleton.remove_undirected(j, k)
            return action_taken

        def rule2(skeleton):
            action_taken = False
            to_be_oriented = []
            for (i,j) in skeleton._directed.keys():
                for k in skeleton.children_of(j):
                    if i in skeleton.neighbors_of(k):
                        to_be_oriented.append((i, k))
                        action_taken = True
            for (i,k) in to_be_oriented:
                skeleton.add_directed(i, k)
                skeleton.remove_undirected(i, k)
            return action_taken

        def rule3(skeleton):
            action_taken = False
            to_be_oriented = []
            for j in skeleton._nodes:
                for i in skeleton.neighbors_of(j):
                    for k in skeleton.parents_of(j):
                        for l in skeleton.parents_of(j):
                            if k != l and k not in skeleton.adjacent_to(l) and  i in skeleton.neighbors_of(k) and i in skeleton.neighbors_of(l):
                                to_be_oriented.append((i, j))
                                action_taken = True
            for (i,j) in to_be_oriented:
                skeleton.add_directed(i, j)
                skeleton.remove_undirected(i, j)
            return action_taken

        if self._representative is None:

            if bool(self.undirected.items()) is False:
                self._representative = copy.deepcopy(self)

            else:

                representative = copy.deepcopy(self)

                while bool(representative.undirected.items()):
                    (i,j), label = list(representative.undirected.items())[0]
                    oriented = False
                    for k in representative.adjacent_to(j):
                        if (k not in representative.adjacent_to(i) and ((k,j) in representative.directed.keys() or (k,j) in representative.bidirected.keys() or (j,k) in representative.bidirected.keys() or (k,j) in representative.semidirected.keys() )):
                            representative.add_directed(j,i,label)
                            representative.remove_undirected(j, i, label)
                            oriented = True
                            break
                    #print((i,j), oriented)
                    if oriented is False:
                        representative.add_directed(i,j,label)
                        representative.remove_undirected(i, j, label)

                    action_taken1 = action_taken2 = action_taken3 = True

                    while (action_taken1 or action_taken2 or action_taken3):
                        action_taken1 = rule1(representative)
                        action_taken2 = rule2(representative)
                        action_taken3 = rule3(representative)
                self._representative = representative

    def get_Markov_blankets(self):
        '''compute the Markov blanket for nodes in DAGs'''
        self.Markov_blankets = {}
        for node in self._nodes:
            self.Markov_blankets[node] = self.children_of(node).union(self.parents_of(node))
            for child in self.children_of(node):
                self.Markov_blankets[node] = self.Markov_blankets[node].union(self.parents_of(child))
                self.Markov_blankets[node].remove(node)


    def BayesBall(self, x,z):

        '''applies the Bayes ball algorithm of networkx to the graph after conversion to nx.graph'''

        G = self.to_nx()

        try:
            x = {x} if x in G else x
            z = {z} if z in G else z

            intersection = x & z
            if intersection:
                raise nx.NetworkXError(
                    f"The sets are not disjoint, with intersection {intersection}"
                )

            set_v = x | z
            if set_v - G.nodes:
                raise nx.NodeNotFound(f"The node(s) {set_v - G.nodes} are not found in G")
        except TypeError:
            raise nx.NodeNotFound("One of x or z is not a node or a set of nodes in G")

        #if not nx.is_directed_acyclic_graph(G):
        #    raise nx.NetworkXError("graph should be directed acyclic")

        # contains -> and <-> edges from starting node T
        forward_deque = deque([])
        forward_visited = set()

        # contains <- and - edges from starting node T
        backward_deque = deque(x)
        backward_visited = set()

        ancestors_or_z = set().union(*[nx.ancestors(G, node) for node in x]) | z | x

        while forward_deque or backward_deque:
            #print(forward_visited)
            #print(backward_visited)
            if backward_deque:
                node = backward_deque.popleft()
                backward_visited.add(node)
                if node in z:
                    continue

                # add <- edges to backward deque
                backward_deque.extend(G.pred[node].keys() - backward_visited)
                # add -> edges to forward deque
                forward_deque.extend(G.succ[node].keys() - forward_visited)

            if forward_deque:
                node = forward_deque.popleft()
                forward_visited.add(node)

                # Consider if -> node <- is opened due to ancestor of node in z
                if node in ancestors_or_z:
                    # add <- edges to backward deque
                    backward_deque.extend(G.pred[node].keys() - backward_visited)
                if node not in z:
                    # add -> edges to forward deque
                    forward_deque.extend(G.succ[node].keys() - forward_visited)

        return self.nodes - x - z - forward_visited - backward_visited

    def get_possible_ancestors(self,node):
        pancestors = set()
        pparents = self.possibleparents_of(node)
        if len(pparents) >0:
            pancestors = pancestors.union(pparents)
            #print(pancestors)
            stop = False
            while stop is False:
                new_additions = set()
                for i in pancestors:
                    new_additions = new_additions.union(self.possibleparents_of(i))
                if node in new_additions:
                    new_additions.remove(node)
                enlarged = copy.deepcopy(pancestors).union(new_additions)
                if enlarged == pancestors:
                    stop = True
                pancestors = enlarged
                #print(pancestors)

        return pancestors

    def get_ancestors(self,node):
        ancestors = set()
        parents = self.parents_of(node)
        if len(parents) >0:
            ancestors = ancestors.union(parents)
            #print(ancestors)
            stop = False
            while stop is False:
                new_additions = set()
                for i in ancestors:
                    #print(i)
                    #print(self.parents_of(i))
                    new_additions = new_additions.union(self.parents_of(i))
                #print(new_additions)
                enlarged = copy.deepcopy(ancestors).union(new_additions)
                if enlarged == ancestors:
                    stop = True
                ancestors = enlarged
                #print(pancestors)
        #self._ancestors[node] = ancestors
        return ancestors

    def is_ancestor_of(self, anc, desc) -> bool:
        """
        Check if ``anc`` is an ancestor of ``desc``

        Return
        ------
        bool
            True if ``anc`` is an ancestor  of ``desc`` """

        return anc in self._parents[desc] or anc in self.get_ancestors(desc)


    def find_v_structures(self):
        v_structures = []
        for i,j,k in itertools.permutations(self._nodes,3):
            if (i,j) in self._directed.keys() or (i,j) in self._semidirected.keys() or (i,j) in self._bidirected.keys():
                if (k, j) in self._directed.keys() or (k, j) in self._semidirected.keys() or (k, j) in self._bidirected.keys():
                    if i not in self.adjacent_to(k) and (k,j,i) not in v_structures:
                        v_structures.append((i,j,k))
        return v_structures

    def get_skeleton(self):
        skeleton = LabelledMixedGraph(nodes=self._nodes)
        for (i,j), label in self._directed.items():
            skeleton.add_undirected(i,j,label)
        for (i,j), label in self._semidirected.items():
            skeleton.add_undirected(i,j,label)
        for (i,j), label in self._bidirected.items():
            skeleton.add_undirected(i,j,label)
        for (i,j), label in self._undirected.items():
            skeleton.add_undirected(i,j,label)

        return skeleton

    def get_CPDAG(self):
        '''Computes the CPDAG of a directed acyclic graph'''

        def insert_v_structures(skeleton,v_structures):
            for (i,j,k) in v_structures:

                skeleton.remove_undirected(i, j)
                skeleton.add_directed(i, j)

                skeleton.remove_undirected(k, j)
                skeleton.add_directed(k, j)

        def rule1(skeleton):
            action_taken = False
            to_be_oriented = []
            for (i, j) in skeleton._directed.keys():
                for k in skeleton.neighbors_of(j):
                    if i not in skeleton.adjacent_to(k):
                        to_be_oriented.append((j,k))
                        action_taken = True
            for (j,k) in to_be_oriented:

                skeleton.remove_undirected(j, k)
                skeleton.add_directed(j, k)
            return action_taken

        def rule2(skeleton):
            action_taken = False
            to_be_oriented = []
            for (i,j) in skeleton._directed.keys():
                for k in skeleton.children_of(j):
                    if i in skeleton.neighbors_of(k):
                        to_be_oriented.append((i, k))
                        action_taken = True
            for (i,k) in to_be_oriented:

                skeleton.remove_undirected(i, k)
                skeleton.add_directed(i, k)
            return action_taken

        def rule3(skeleton):
            action_taken = False
            to_be_oriented = []
            for j in skeleton._nodes:
                for i in skeleton.neighbors_of(j):
                    for k in skeleton.parents_of(j):
                        for l in skeleton.parents_of(j):
                            if k != l and k not in skeleton.adjacent_to(l) and  i in skeleton.neighbors_of(k) and i in skeleton.neighbors_of(l):
                                to_be_oriented.append((i, j))
                                action_taken = True
            for (i,j) in to_be_oriented:

                skeleton.remove_undirected(i, j)
                skeleton.add_directed(i, j)
            return action_taken


        if (self.num_bidirected > 0 or self.num_semidirected > 0 or self.num_undirected > 0):
            raise ValueError(
                'CPDAGs can only be computed for DAGs')

        else:
            skeleton = self.get_skeleton()
            v_structures = self.find_v_structures()

            insert_v_structures(skeleton,v_structures)

            action_taken1 = action_taken2 = action_taken3 = True

            while(action_taken1 or action_taken2 or action_taken3):
                action_taken1 = rule1(skeleton)
                action_taken2 = rule2(skeleton)
                action_taken3 = rule3(skeleton)

            return skeleton

    def is_d_separated(self, x, y, z, DAG_check = True):
        """Return whether node sets `x` and `y` are d-separated by `z` using the networkx implemention.
         An additional Boolean parameter DAG_check has been added that can disable the networkx check for acyclicity as this can be
         a computational bottleneck.

        Parameters
        ----------
        G : nx.DiGraph
            A NetworkX DAG.

        x : node or set of nodes
            First node or set of nodes in `G`.

        y : node or set of nodes
            Second node or set of nodes in `G`.

        z : node or set of nodes
            Potential separator (set of conditioning nodes in `G`). Can be empty set.

        DAG_check:  bool
                    If True the networkx function is_acyclic() is employed to verify acyclicity.
                    If False no such check is executed.

        Returns
        -------
        b : bool
            A boolean that is true if `x` is d-separated from `y` given `z` in `G`.

        Raises
        ------
        NetworkXError
            The *d-separation* test is commonly used on disjoint sets of
            nodes in acyclic directed graphs.  Accordingly, the algorithm
            raises a :exc:`NetworkXError` if the node sets are not
            disjoint or if the input graph is not a DAG.

        NodeNotFound
            If any of the input nodes are not found in the graph,
            a :exc:`NodeNotFound` exception is raised

        Notes
        -----
        A d-separating set in a DAG is a set of nodes that
        blocks all paths between the two sets. Nodes in `z`
        block a path if they are part of the path and are not a collider,
        or a descendant of a collider. Also colliders that are not in `z`
        block a path. A collider structure along a path
        is ``... -> c <- ...`` where ``c`` is the collider node.

        https://en.wikipedia.org/wiki/Bayesian_network#d-separation
        """

        if self.nx_graph is None:
            self.nx_graph = self.to_nx()

        G = self.nx_graph




        try:
            x = {x} if x in G else x
            y = {y} if y in G else y
            z = {z} if z in G else z

            intersection = x & y or x & z or y & z
            if intersection:
                raise nx.NetworkXError(
                    f"The sets are not disjoint, with intersection {intersection}"
                )

            set_v = x | y | z
            if set_v - G.nodes:
                raise nx.NodeNotFound(f"The node(s) {set_v - G.nodes} are not found in G")
        except TypeError:
            raise nx.NodeNotFound("One of x, y, or z is not a node or a set of nodes in G")

        if self.DAG is None and DAG_check is True:
            self.DAG = nx.is_directed_acyclic_graph(G)

        if self.DAG is False:
            raise nx.NetworkXError("graph should be directed acyclic")

        # contains -> and <-> edges from starting node T
        forward_deque = deque([])
        forward_visited = set()

        # contains <- and - edges from starting node T
        backward_deque = deque(x)
        backward_visited = set()

        ancestors_or_z = set().union(*[self.get_ancestors(node) for node in x]) | z | x

        while forward_deque or backward_deque:
            if backward_deque:
                node = backward_deque.popleft()
                backward_visited.add(node)
                if node in y:
                    return False
                if node in z:
                    continue

                # add <- edges to backward deque
                backward_deque.extend(G.pred[node].keys() - backward_visited)
                # add -> edges to forward deque
                forward_deque.extend(G.succ[node].keys() - forward_visited)

            if forward_deque:
                node = forward_deque.popleft()
                forward_visited.add(node)
                if node in y:
                    return False

                # Consider if -> node <- is opened due to ancestor of node in z
                if node in ancestors_or_z:
                    # add <- edges to backward deque
                    backward_deque.extend(G.pred[node].keys() - backward_visited)
                if node not in z:
                    # add -> edges to forward deque
                    forward_deque.extend(G.succ[node].keys() - forward_visited)

        return True

    def find_minimal_d_separator(self, x, y, *, included=None, restricted=None, DAG_check = True):
        """Returns a minimal d-separating set between `x` and `y` if possible, using the networkx implemention.
         An additional parameter DAG_check has been added which can disable the networkx check for acyclicity as this can be
         a computational bottleneck.

        A d-separating set in a DAG is a set of nodes that blocks all
        paths between the two sets of nodes, `x` and `y`. This function
        constructs a d-separating set that is "minimal", meaning no nodes can
        be removed without it losing the d-separating property for `x` and `y`.
        If no d-separating sets exist for `x` and `y`, this returns `None`.

        In a DAG there may be more than one minimal d-separator between two
        sets of nodes. Minimal d-separators are not always unique. This function
        returns one minimal d-separator, or `None` if no d-separator exists.

        Uses the algorithm presented in [1]_. The complexity of the algorithm
        is :math:`O(m)`, where :math:`m` stands for the number of edges in
        the subgraph of G consisting of only the ancestors of `x` and `y`.
        For full details, see [1]_.

        Parameters
        ----------
        G : graph
            A networkx DAG.
        x : set | node
            A node or set of nodes in the graph.
        y : set | node
            A node or set of nodes in the graph.
        included : set | node | None
            A node or set of nodes which must be included in the found separating set,
            default is None, which means the empty set.
        restricted : set | node | None
            Restricted node or set of nodes to consider. Only these nodes can be in
            the found separating set, default is None meaning all nodes in ``G``.
        DAG_check:  bool
                    If True the networkx function is_acyclic() is employed to verify acyclicity.
                    If False no such check is executed.

        Returns
        -------
        z : set | None
            The minimal d-separating set, if at least one d-separating set exists,
            otherwise None.

        Raises
        ------
        NetworkXError
            Raises a :exc:`NetworkXError` if the input graph is not a DAG
            or if node sets `x`, `y`, and `included` are not disjoint.

        NodeNotFound
            If any of the input nodes are not found in the graph,
            a :exc:`NodeNotFound` exception is raised.

        References
        ----------
        .. [1] van der Zander, Benito, and Maciej Liśkiewicz. "Finding
            minimal d-separators in linear time and applications." In
            Uncertainty in Artificial Intelligence, pp. 637-647. PMLR, 2020.
        """

        def _reachable(G, x, a, z):
            """Modified Bayes-Ball algorithm for finding d-connected nodes.

            Find all nodes in `a` that are d-connected to those in `x` by
            those in `z`. This is an implementation of the function
            `REACHABLE` in [1]_ (which is itself a modification of the
            Bayes-Ball algorithm [2]_) when restricted to DAGs.

            Parameters
            ----------
            G : nx.DiGraph
                A NetworkX DAG.
            x : node | set
                A node in the DAG, or a set of nodes.
            a : node | set
                A (set of) node(s) in the DAG containing the ancestors of `x`.
            z : node | set
                The node or set of nodes conditioned on when checking d-connectedness.

            Returns
            -------
            w : set
                The closure of `x` in `a` with respect to d-connectedness
                given `z`.

            References
            ----------
            .. [1] van der Zander, Benito, and Maciej Liśkiewicz. "Finding
                minimal d-separators in linear time and applications." In
                Uncertainty in Artificial Intelligence, pp. 637-647. PMLR, 2020.

            .. [2] Shachter, Ross D. "Bayes-ball: The rational pastime
               (for determining irrelevance and requisite information in
               belief networks and influence diagrams)." In Proceedings of the
               Fourteenth Conference on Uncertainty in Artificial Intelligence
               (UAI), (pp. 480–487). 1998.
            """

            def _pass(e, v, f, n):
                """Whether a ball entering node `v` along edge `e` passes to `n` along `f`.

                Boolean function defined on page 6 of [1]_.

                Parameters
                ----------
                e : bool
                    Directed edge by which the ball got to node `v`; `True` iff directed into `v`.
                v : node
                    Node where the ball is.
                f : bool
                    Directed edge connecting nodes `v` and `n`; `True` iff directed `n`.
                n : node
                    Checking whether the ball passes to this node.

                Returns
                -------
                b : bool
                    Whether the ball passes or not.

                References
                ----------
                .. [1] van der Zander, Benito, and Maciej Liśkiewicz. "Finding
                   minimal d-separators in linear time and applications." In
                   Uncertainty in Artificial Intelligence, pp. 637-647. PMLR, 2020.
                """
                is_element_of_A = n in a
                # almost_definite_status = True  # always true for DAGs; not so for RCGs
                collider_if_in_Z = v not in z or (e and not f)
                return is_element_of_A and collider_if_in_Z  # and almost_definite_status

            queue = deque([])
            for node in x:
                if bool(G.pred[node]):
                    queue.append((True, node))
                if bool(G.succ[node]):
                    queue.append((False, node))
            processed = queue.copy()

            while any(queue):
                e, v = queue.popleft()
                preds = ((False, n) for n in G.pred[v])
                succs = ((True, n) for n in G.succ[v])
                f_n_pairs = chain(preds, succs)
                for f, n in f_n_pairs:
                    if (f, n) not in processed and _pass(e, v, f, n):
                        queue.append((f, n))
                        processed.append((f, n))

            return {w for (_, w) in processed}

        if self.nx_graph is None:
            self.nx_graph = self.to_nx()

        G = self.nx_graph

        if self.DAG is None and DAG_check is True:
            self.DAG = nx.is_directed_acyclic_graph(G)

        if self.DAG is False:
            raise nx.NetworkXError("graph should be directed acyclic")

        try:
            x = {x} if x in G else x
            y = {y} if y in G else y

            if included is None:
                included = set()
            elif included in G:
                included = {included}

            if restricted is None:
                restricted = set(G)
            elif restricted in G:
                restricted = {restricted}

            set_y = x | y | included | restricted
            if set_y - G.nodes:
                raise nx.NodeNotFound(f"The node(s) {set_y - G.nodes} are not found in G")
        except TypeError:
            raise nx.NodeNotFound(
                "One of x, y, included or restricted is not a node or set of nodes in G"
            )

        if not included <= restricted:
            raise nx.NetworkXError(
                f"Included nodes {included} must be in restricted nodes {restricted}"
            )

        intersection = x & y or x & included or y & included
        if intersection:
            raise nx.NetworkXError(
                f"The sets x, y, included are not disjoint. Overlap: {intersection}"
            )

        nodeset = x | y | included
        ancestors_x_y_included = nodeset.union(*[self.get_ancestors(node) for node in nodeset])

        z_init = restricted & (ancestors_x_y_included - (x | y))

        x_closure = _reachable(G, x, ancestors_x_y_included, z_init)
        if x_closure & y:
            return None

        z_updated = z_init & (x_closure | included)
        y_closure = _reachable(G, y, ancestors_x_y_included, z_updated)
        return z_updated & (y_closure | included)


    # def to_MAG(self, latent_nodes):
    #
    #     def powerset(s, r_min=0, r_max=None):
    #         if r_max is None: r_max = len(s)
    #         return map(set, itertools.chain(*(itertools.combinations(s, r) for r in range(r_min, r_max + 1))))
    #
    #     new_nodes = self._nodes - latent_nodes
    #     directed = set()
    #     bidirected = set()
    #     for i, j in itertools.combinations(self._nodes - latent_nodes, r=2):
    #         adjacent = all(not self.is_d_separated(i, j, S) for S in powerset(self._nodes - {i, j} - latent_nodes))
    #         if adjacent:
    #             if self.is_ancestor_of(i, j):
    #                 directed.add((i, j))
    #             elif self.is_ancestor_of(j, i):
    #                 directed.add((j, i))
    #             else:
    #                 bidirected.add((i, j))
    #
    #     directed_with_label = {(i,j): None for (i,j) in directed}
    #     bidirected_with_label = {frozenset({i,j}): None for (i,j) in bidirected}
    #
    #     #if relabel is not None:
    #     #    t = self.topological_sort()
    #     #    t_new = [node for node in t if node not in latent_nodes]
    #     #    node2new_label = dict(map(reversed, enumerate(t_new)))
    #     #    new_nodes = {node2new_label[node] for node in new_nodes}
    #     #    directed = {(node2new_label[i], node2new_label[j]) for i, j in directed}
    #     #    bidirected = {(node2new_label[i], node2new_label[j]) for i, j in bidirected}
    #
    #     return LabelledMixedGraph(nodes=new_nodes, directed=directed_with_label, bidirected=bidirected_with_label)