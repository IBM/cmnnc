# Copyright (c) 2019-2020, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import typing
import dataclasses as dc
from pprint import pprint

import numpy as np
import onnxruntime as onnxrt
import onnx

import conv
import pipeline as pl
from op_info import OpInfo, OpInfo_CONV, OpInfo_ADD
from onnx_test_models import mk_simple_residual as onnx_mk_simple_residual
from onnx_util import onnx_rand_input, onnx_conv_get_params, onnx_conv_get_batch, \
                      onnx_get_obj_shapes, NodeId, EdgeName, onnx_get_ins_outs


# TODO: move this to another file when done

def get_obj_shapes_nobatch(g):
    """ Get object shapes for a graph, but ignore batch parameter """
    obj_shapes = onnx_get_obj_shapes(g)
    for objname in obj_shapes:
        shape = obj_shapes[objname]
        assert shape[0] == 1 # batch size
        obj_shapes[objname] = shape[1:]
    return obj_shapes

@dc.dataclass(init=False)
class Partition:
    """ A partition of (ONNX) nodes

    This class is refined in steps:
     * __init__()
     * set_conv_ps(): set the parameters for the convlution of this partition.
    """
    nis: typing.List[NodeId] # nodes that belong to this partition

    def __init__(self, *nis):
        self.nis = list(nis)

    def set_conv_ps(self, conv_ps):
        self.conv_ps_ = conv_ps

class OnnxGraph:
    """ This class wraps an ONNX module so that it can be used for our purposes """
    m: onnx.ModelProto

    # NB: An edge might be an input to multiple nodes,
    # but it can only be the output of a single node
    inps: typing.Dict[EdgeName, typing.List[NodeId]]
    outs: typing.Dict[EdgeName, NodeId]
    partitions: typing.List[Partition]

    objs_info: typing.Dict[str, pl.ObjectInfo]

    # dicts for easy lookip
    # Value info for graph input, output, and intermediate valudes
    input_vis: typing.Dict[str, onnx.helper.ValueInfoProto]
    output_vis: typing.Dict[str, onnx.helper.ValueInfoProto]
    inter_vis: typing.Dict[str, onnx.helper.ValueInfoProto]
    # initializer tensor values
    init_tvs: typing.Dict[str, onnx.helper.TensorProto]

    @property
    def g(self):
        """ shortcut for onnx graph """
        return self.m.graph

    def __init__(self, onnx_m):
        self.m = onnx_m

        (self.inps, self.outs) = onnx_get_ins_outs(self.g)

        self.partitions = self.partition_()

        self.input_vis =  dict((e.name,e) for e in self.g.input)
        self.output_vis = dict((e.name,e) for e in self.g.output)
        self.inter_vis = dict((e.name,e) for e in self.g.value_info)
        self.init_tvs = dict((v.name,v) for v in self.g.initializer)

        self.objs_info = dict(
            (name, pl.ObjectInfo(shape))
                for (name, shape) in get_obj_shapes_nobatch(self.g).items()
        )

        self.process_partitions()

    def get_value_info(self, e: str):
        for d in (self.input_vis, self.output_vis, self.inter_vis):
            if e in d:
                return d[e]
        raise ValueError("No value info for %s" % (e,))

    def get_value_shape(self, e: str) -> typing.Tuple[int,...]:
        shape = self.get_value_info(e).type.tensor_type.shape
        return tuple(x.dim_value for x in shape.dim)

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

    def partition_(self) -> typing.List[Partition]:
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
                raise ValueError("Unknown node type %s" % (node.op_type, ))

        partlist.append(part)
        return partlist

    def process_partitions_1(self):
        init_tvs = self.init_tvs
        # Pass 1: set conv_ps and objs_info
        for part in self.partitions:
            ni0 = part.nis[0]
            node0 = self.g.node[ni0]
            if node0.op_type != 'Conv':
                raise ValueError("First partition node is not Conv (%s)" % (node0.op_type,))
            # input name for the convolution node
            # set conv_ps for partition
            conv_ps = onnx_conv_get_params(self.g, node0)
            part.conv_ps = conv_ps
            # set padding to the input node
            (input_name,)   = (x for x in node0.input if x not in init_tvs)
            self.objs_info[input_name].padding = part.conv_ps.get_input_padding()

    def process_partitions_2(self):
        # Pass 2: set stage info
        init_tvs = self.init_tvs
        for (pid, part) in enumerate(self.partitions):
            opinfos = []
            for (idx,ni) in enumerate(part.nis):
                oi = None
                node = self.g.node[ni]
                if idx == 0: # first node!
                    assert getattr(part, "conv_domain", None) is None
                    assert node.op_type == 'Conv'
                    (input_name,)   = (x for x in node.input if x not in init_tvs)
                    (weights_name,) = (x for x in node.input if x in init_tvs)
                    (output_name,)  = node.output

                    # TODO: The ONNX Conv operator has a batch size. Seems to
                    # me that the best thing to do would be to deal with batch
                    # size externally (i.e., at an external loop that performs
                    # the transfers from/to the accelerator), and transform the
                    # ONNX nodes to have a batch size of 1.
                    assert onnx_conv_get_batch(self.g, node) == 1

                    out_padding = self.objs_info[output_name].padding
                    part.conv_ps.set_p_out_from_padding(out_padding)

                    oi = OpInfo_CONV(part.conv_ps,
                                       s_id="S%d" % (pid,),
                                       vin_id=input_name,
                                       vout_id=output_name)
                    part.conv_domain = oi.get_domain()

                elif node.op_type == 'Add':
                    assert part.conv_domain is not None
                    (in1, in2) = node.input
                    (out,) =  node.output
                    a_shape = self.get_value_shape(in1)
                    b_shape = self.get_value_shape(in2)
                    y_shape = self.get_value_shape(out)
                    assert a_shape == b_shape
                    assert a_shape == y_shape
                    oi = OpInfo_ADD(part.conv_domain, a_shape, in1, in2, out)
                elif node.op_type == 'Conv':
                    raise ValueError("Conv node can only be the first in a partition")
                else:
                    raise ValueError("Unknown node type %s" % (node.op_type, ))

                assert oi is not None
                opinfos.append(oi)
            part.stage_info = pl.StageInfo(opinfos)

    def process_partitions_3(self):
        # Pass 3: set core_conf
        init_tvs = self.init_tvs
        for (pid, part) in enumerate(self.partitions):
            conv_ni = self.partitions[pid].nis[0]
            conv_node = self.g.node[conv_ni]
            (weights_name,) = (x for x in conv_node.input if x in init_tvs)

            part.core_conf = pl.CoreConf(
                np.array(init_tvs[weights_name].float_data)
                  .reshape(part.conv_ps.eval("(f.l, f.d*f.h*f.w)"))
            )

    def process_partitions(self):
        self.process_partitions_1()
        self.process_partitions_2()
        self.process_partitions_3()

    def get_stage_info(self, pid) -> pl.StageInfo:
        return self.partitions[pid].stage_info

    def get_core_conf(self, pid) -> pl.CoreConf:
        return self.partitions[pid].core_conf

def test_onnx_residual_2d():
    # Create the following ONNX graph
    # (this is what onnx_mk_simple_residual does)
    #
    #  CONV2D ---> CONV2D ---> ADD
    #          |                ^
    #          |                |
    #          +--------------- +
    #
    # CONV2D
    #   input:  in
    #   output: v1
    #   weights: w1
    # CONV2D
    #   input:  v1
    #   output: v2
    #   weights: w2
    # ADD
    #  input: v1,v2
    #  output: out
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

    # Parse onnx graph to create a pipeline
    graph = OnnxGraph(onnx_m)
    pprint(graph.partitions)

    vals = {}
    stages = []
    cconfs = []
    for pid in range(len(graph.partitions)):
        si = graph.get_stage_info(pid)
        stage = pl.Stage(si, vals)
        stages.append(stage)
        cconf = graph.get_core_conf(pid)
        cconfs.append(cconf)

    pline = pl.Pipeline(stages, graph.objs_info, execute_ops = True, loop_inp_limit=1)
    pline.configure(cconfs)

    # set inputs
    inp = onnx_rand_input(onnx_m)
    for (inp_name, inp_data) in inp.items():
        obj_info = graph.objs_info[inp_name]
        assert inp_data.shape == (1,) + obj_info.shape # NB: batching
        # data = np.random.rand(*obj_info.shape)
        data = inp_data[0]
        data = np.pad(data, obj_info.padding)
        obj = pline.get_object(inp_name)
        obj[...] = data

    # Execute the pipeline
    print_info = False
    for iters in pline.tick_gen():
        if print_info:
            print("*"*80)
        for (s,i) in iters.items():
            if print_info:
                print("%s: %s" % (s, i))
        if print_info:
            print("*"*80)
    print("%s> DONE" % ("-"*30,))

    # Get pipeline results
    pline_out = pline.get_object('out')
    pline_v1 = pline.get_object('v1')

    # Execute using onnxruntime
    onnx.save(onnx_m, 'simple_residual_2d.onnx')
    sess = onnxrt.InferenceSession('simple_residual_2d.onnx')
    out = sess.run(None, inp)

    # Execute using manual ops
    in_m = np.pad(inp['in'][0], graph.objs_info['in'].padding)
    w1_m = np.array(graph.init_tvs["w1"].float_data).reshape(conv1_ps.get_filters_shape())
    v1_m = conv.conv2d_simple(in_m, w1_m, conv1_ps)
    v1_m = np.pad(v1_m, graph.objs_info['v1'].padding)
    np.testing.assert_allclose(v1_m, pline_v1, err_msg="pipeline v1  does not match manual v1")

    np.testing.assert_allclose(out[0][0,:], pline_out, err_msg="OUT does not match")

    return graph

if __name__ == '__main__':
    ret = test_onnx_residual_2d()
