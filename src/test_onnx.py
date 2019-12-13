# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import typing
import dataclasses as dc
from pprint import pprint

import onnxruntime as onnxrt
import onnx

import conv
from pipeline import StageInfo
from onnx_test_models import mk_simple_residual as onnx_mk_simple_residual
from onnx_util import onnx_rand_in



# TODO: move this to another file when done

NodeId = int
EdgeName = str

@dc.dataclass(init=False)
class Partition:
    nis: typing.List[NodeId] # nodes that belong to this partition

    def __init__(self, *nis):
        self.nis = list(nis)


class OnnxGraph:
    """ This class wraps an ONNX module so that it can be used for our purposes """
    m: onnx.ModelProto
    # NB: A given edge might be an input to multiple nods, but it can only
    # be the output of a single node
    inps: typing.Dict[EdgeName, typing.List[NodeId]]
    outs: typing.Dict[EdgeName, NodeId]

    def __init__(self, onnx_m):
        self.m = onnx_m

        self.inps = {}
        self.outs = {}
        for (nid, node) in enumerate(self.g.node):
            for e_in in node.input:
                if e_in not in self.inps:
                    self.inps[e_in] = []
                self.inps[e_in].append(nid)
            del e_in

            for e_out in node.output:
                assert e_out not in self.outs, "Edge %s is output of multiple nodes: %d, %d" % (e_out, self.outs[e_out], nid)
                self.outs[e_out] = nid
            del e_out

        self.partitions = self.partition()

    @property
    def g(self):
        """ shortcut for onnx graph """
        return self.m.graph

    def get_src_nis(self) -> typing.Iterator[NodeId]:
        """ Return source node ids

        Source nodes are nodes with inputs which are either part of the graph
        input or part of the graph initializer """
        inputs = set((x.name for x in self.g.input))
        initializers = set((x.name for x in self.g.initializer))
        non_internal = inputs.union(initializers)

        for (i,n) in enumerate(self.g.node):
            n_inputs = set(n.input)
            diff = n_inputs.difference(non_internal)
            if len(diff) == 0:
                yield i

    def get_out_nis(self, nid: NodeId) -> typing.Iterator[NodeId]:
        """ Returns the node ids that have (write) dependencies on this node """
        node = self.g.node[nid]
        # for every output edge, find and yield the nodes for which it is an
        # input
        for out in node.output:
            for nid in self.inps.get(out, []):
                yield nid

    def topsort(self) -> typing.List[NodeId]:
        """ Topological sort of the node ids """
        # https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
        white = set(range(len(self.g.node)))
        gray  = set()
        black = set()

        ret = []
        def visit(nid: NodeId):
            if nid in black:
                return
            if nid in gray:
                raise RuntimeError("Loop detected. Graph is not a DAG.")

            white.remove(nid)
            gray.add(nid)
            for o_nid in self.get_out_nis(nid):
                visit(o_nid)
            gray.remove(nid)
            black.add(nid)
            ret.append(nid)

        while len(white) > 0:
            n = next(iter(white))
            visit(n)

        return list(reversed(ret))

    def partition(self) -> typing.List[Partition]:
        # Partition the ONNX graph, so that we can map each partition to a core
        #
        # Invariants:
        #  1. one CONV per partition (we can revisit this later)
        #  2. No cycles in the partition graph (we can also revisit later)
        #
        # [CONV1] ---> [CONV2] ---> [ADD1]
        #    |                        ^
        #    |                        |
        #    +------------------------+
        #
        # That is, in the above example, cannot partition as:
        #  P1: [CONV1, ADD1]
        #  P2: [CONV2]
        # because this would create a cycle in the partition graph
        #
        # If we sort the nodes in a topological order, then we can just bind
        # every non-conv node with its previous conv.
        #
        tslist = self.topsort()
        part = Partition(tslist[0])
        partlist = []
        for nid in tslist[1:]:
            node = self.g.node[nid]
            if node.op_type == 'Conv':
                partlist.append(part)
                part = Partition(nid)
            elif node.op_type in ('Add',):
                part.nis.append(nid)
            else:
                raise ValueError("Uknown node type %s" % (node.op_type, ))

        partlist.append(part)
        return partlist

def onnx_partition(onnx_model) -> typing.List[StageInfo]:
    pass

def test_onnx_residual_2d():
    # Create the following ONNX graph:
    #
    #  CONV2D ---> CONV2D ---> ADD
    #          |                ^
    #          |                |
    #          +--------------- +
    #
    conv1_padding = 1
    conv2_padding = 1

    conv1_ps = conv.Conv2DParams(
        i = conv.Conv2DInParams(w=32, h=32, d=3),
        f = conv.Conv2DFiltParams(w=3, h=3, d=3, l=1),
        p = conv1_padding,
        p_out = conv2_padding,
        s = 1)

    conv2_ps = conv.Conv2DParams(
        i = conv1_ps.o.to_in(),
        f = conv.Conv2DFiltParams(w=3, h=3, d=conv1_ps.f.l, l=1),
        p = conv2_padding,
        p_out = 0,
        s = 1)

    onnx_m = onnx_mk_simple_residual(conv1_ps, conv2_ps)
    ret = onnx_m
    inp = onnx_rand_in(onnx_m)

    # Parse onnx graph to create a pipeline
    graph = OnnxGraph(onnx_m)

    # Configure the pipeline based using the ONNX initializers

    # Execute the pipeline
    # image  = np.pad(inp["in"][0], self.conv_ps.get_padding())

    # Execute using onnxruntime
    onnx.save(onnx_m, 'simple_residual_2d.onnx')
    sess = onnxrt.InferenceSession('simple_residual_2d.onnx')
    out = sess.run(None, inp)

    return graph

if __name__ == '__main__':
    ret = test_onnx_residual_2d()
