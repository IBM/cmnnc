# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

from itertools import chain
import typing

import onnx
import numpy as np

import conv

# ONNX helpers

NodeId = int
EdgeName = str

def onnx_get_init_data(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return init

def onnx_rand_input(model: onnx.ModelProto):
    """ Create random inputs for a given ONNX model """
    ret = {}
    for inp in model.graph.input:
        tensor_ty = inp.type.tensor_type
        elem_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_ty.elem_type]
        shape = [d.dim_value for d in tensor_ty.shape.dim]
        ret[inp.name] = np.random.random(shape).astype(elem_type)
        # print(inp.name)
    return ret

def onnx_conv_get_batch(graph: onnx.GraphProto, node) -> int:
    """ Get the batch size of an ONNX Conv node """
    if node.op_type != 'Conv':
        raise TypeError("Expecting type 'Conv', but got type:'%s'" (node.op_type,))

    # Input is not in the initializer data, while weights are
    init_names = set(x.name for x in graph.initializer)
    (input_name,) = (x for x in node.input if x not in init_names)

    # Try to find the input in the inputs or value info part of the graph
    for vi in chain(graph.value_info, graph.input):
        if vi.name == input_name:
            inp = vi
            break
    else: raise AssertionError("Did not find input. Bailing out")

    batch_size = inp.type.tensor_type.shape.dim[0].dim_value
    return batch_size

def onnx_get_obj_shapes(graph: onnx.GraphProto):
    ret = {}
    for vi in chain(graph.value_info, graph.input, graph.output):
        ret[vi.name] = tuple(x.dim_value for x in vi.type.tensor_type.shape.dim)
    return ret

def onnx_conv_get_params(graph: onnx.GraphProto, node):
    """ Create a Conv2DParams structure from an ONNX Conv node """
    if node.op_type != 'Conv':
        raise TypeError("Expecting type 'Conv', but got type:'%s'" (node.op_type,))
    attrs = dict( (x.name,x) for x in node.attribute )

    # Padding: for now, fail if there are different padding for different
    # dimensions
    pads = attrs['pads'].ints
    p = pads[0]
    if not all([p == x for x in pads[1:]]):
        raise NotImplementedError("pads: %s not supported" % (pads,))

    # filter size is a bit tricky.
    # One option might be kernel_shape, but this does not include the full
    # information (e.g., it can be 3x3 which does not include the number
    # of layers. Instead, we use the weights to defer this information.

    # Input is not in the initializer data, while weights are
    init_names = set(x.name for x in graph.initializer)
    (input_name,) = (x for x in node.input if x not in init_names)
    (weights_name,) = (x for x in node.input if x in init_names)

    # Try to find the input in the inputs or value info part of the graph
    for vi in chain(graph.value_info, graph.input):
        if vi.name == input_name:
            inp = vi
            break
    else: raise AssertionError("Did not find input. Bailing out")

    # Try to find the weights in the initializer part of the graph
    for vi in graph.initializer:
        if vi.name == weights_name:
            weights = vi
            break
    else: raise AssertionError("Did not find weights in initalizer data. Bailing out")

    # https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv:
    #    The weight tensor that will be used in the convolutions; has size (M x
    #    C/group x kH x kW), where C is the number of channels, and kH and kW
    #    are the height and width of the kernel, and M is the number of feature
    #    maps.  For more than 2 dimensions, the kernel shape will be (M x
    #    C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the
    #    dimension of the kernel.  (...)
    f = conv.Conv2DFiltParams(
        w=weights.dims[-1],
        h=weights.dims[-2],
        d=weights.dims[-3],
        l=weights.dims[-4]
    )

    # https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv:
    #    Input data tensor from previous layer; has size (N x C x H x W), where
    #    N is the batch size, C is the number of channels, and H and W are the
    #    height and width. Note that this is for the 2D image. Otherwise the
    #    size is (N x C x D1 x D2 ... x Dn). Optionally, if dimension
    #    denotation is in effect, the operation expects input data tensor to
    #    arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL,
    #    DATA_FEATURE, DATA_FEATURE ...].
    # NB: We ignore the batch size
    i = conv.Conv2DInParams(
        w = inp.type.tensor_type.shape.dim[-1].dim_value,
        h = inp.type.tensor_type.shape.dim[-2].dim_value,
        d = inp.type.tensor_type.shape.dim[-3].dim_value
    )

    conv_ps = conv.Conv2DParams(
        i = i,
        f = f,
        p = p,
        # TODO: deal with strides
        s = 1,
        p_out = 0
    )

    #print("%s" % (conv_ps,))
    return conv_ps

def onnx_get_ins_outs(graph: onnx.GraphProto) \
    -> typing.Tuple[typing.Dict[EdgeName, typing.List[NodeId]],
                    typing.Dict[EdgeName, NodeId]]:

        ret = (inps, outs) = ({},{})
        for (nid, node) in enumerate(graph.node):
            for e_in in node.input:
                if e_in not in inps:
                    inps[e_in] = []
                inps[e_in].append(nid)
            del e_in

            for e_out in node.output:
                assert e_out not in outs, "Edge %s is output of multiple nodes: %d, %d" % (e_out, outs[e_out], nid)
                outs[e_out] = nid
            del e_out
        return ret
