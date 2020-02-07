# Author: Kornilios Kourtis <kornilios@gmail.com`>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4 nowrap:

import typing
import itertools

import graphviz
from z3 import z3

""" Map a graph to another using Z3

This represents the problem of mapping a dataflow graph to the hardware graph.
This is similar to graph isomorphism but it only works on one direction.
"""

class Graph(object):
    """ A graph (essentially a set of edges) """
    name_: str
    edges_: typing.Set[typing.Tuple[str,str]]

    def __init__(self, name):
        self.name_  = name
        self.edges_ = set()

    def nodes(self) -> typing.Set[str]:
        ret = set()
        for (n1,n2) in self.edges_:
            ret.add(n1)
            ret.add(n2)
        return ret

    def edges(self):
        return list(self.edges_)

    def has_edge(self, n1: str, n2: str):
        return (n1,n2) in self.edges_

    def add_edge(self, n1: str, n2: str):
        if n1 == n2:
            raise ValueError("Cannot add self-edge: %s->%s" % (n1,n2))
        self.edges_.add((n1,n2))

    def del_edge(self, n1: str, n2: str):
        if (n1,n2) not in self.edge_:
            raise ValueError("Edge: %s->%s does not exist" % (n1,n2))
        self.edges_.remove((n1,n2))

class GraphZ3(object):
    def __init__(self, s: z3.Solver, g: Graph):
        (G, G_nodes) = z3.EnumSort(g.name_, tuple(g.nodes()))
        # Function that returns true if there is an edge between two nodes, 
        # and false otherwise
        edges_f = z3.Function('G_%s_edges' % (g.name_,), G, G, z3.BoolSort())
        for (n1,n2) in itertools.product(iter(G_nodes), iter(G_nodes)):
            n1_str = repr(n1)
            n2_str = repr(n2)
            s.add(edges_f(n1,n2) == g.has_edge(n1_str,n2_str))

        self.nodes_s = G       # Nodes Sort
        self.nodes   = G_nodes
        self.edges_f = edges_f

def z3_graph_mapping(s: z3.Solver, g1: GraphZ3, g2: GraphZ3):
    map_f = z3.Function('Map', g1.nodes_s, g2.nodes_s)
    x = z3.Const('x', g1.nodes_s)
    y = z3.Const('y', g1.nodes_s)
    s.add([
        z3.ForAll([x,y], g1.edges_f(x,y) == g2.edges_f(map_f(x), map_f(y)))
    ])

    # Check if can satisfy the specified constraints
    if s.check():
        # If can, get the model (i.e., a solution that satisfies the
        # constrains) for the Map function and return it
        m = s.model()
        for d in m.decls():
            if d.name() == 'Map':
                    iso_m = m[d]
                    return iso_m
        raise RuntimeError("Isomorphism function not found in model")
    return None

def graph_mapping(g1: Graph, g2: Graph):
    s = z3.Solver()
    g1_z3 = GraphZ3(s, g1)
    g2_z3 = GraphZ3(s, g2)
    return z3_graph_mapping(s, g1_z3, g2_z3)

class Z3FuncInterpWrapper:
    """ Wrap a Z3 FuncInterp to access its mapping

    This takes a z3.FuncInterp as: [C -> a, A -> c, else -> b]
    And wraps it in a dict so that __call__() can be used to access the mapping.
    """

    def __init__(self, fi: z3.FuncInterp):
        l = fi.as_list()
        # Last element is the default value
        self.d_ = dict()
        self.default_ = l.pop()
        for (k,v) in l:
            assert isinstance(k, z3.DatatypeRef)
            assert isinstance(v, z3.DatatypeRef)
            self.d_[str(k)] = str(v)

    def __call__(self, k):
        return self.d_.get(k, self.default_)

def draw_mapping(fname: str, g1: Graph, g2: Graph, map_fi: z3.FuncInterp):
    map_fiw = Z3FuncInterpWrapper(map_fi)
    for n in g1.nodes():
        print("Node %s mapped to %s" % (n, map_fiw(n)))

    gv = graphviz.Digraph(format="png")

    with gv.subgraph(name="cluster_1") as gv1:
        for (x1,x2) in g1.edges():
            gv1.edge(x1,x2)

    with gv.subgraph(name="cluster_2") as gv2:
        for (x1,x2) in g2.edges():
            gv2.edge(x1,x2)

    for n in g1.nodes():
        gv.edge(repr(n), repr(map_fiw(n)),style="dashed")

    gv.render(fname, view=False, format="pdf")

def test1():
    g1 = Graph("g1")
    g1.add_edge("A", "B")
    g1.add_edge("B", "C")

    g2 = Graph("g2")
    g2.add_edge("c", "b")
    g2.add_edge("b", "a")

    fi = graph_mapping(g1, g2)
    draw_mapping("gmap1", g1, g2, fi)

if __name__ == '__main__':
    test1()
