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
from pipeline import StageInfo

# ONNX helpers

def onnx_get_init_data(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return init

def onnx_rand_in(model):
    """ Create random inputs for a given ONNX model """
    ret = {}
    for inp in model.graph.input:
        tensor_ty = inp.type.tensor_type
        elem_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_ty.elem_type]
        shape = [d.dim_value for d in tensor_ty.shape.dim]
        ret[inp.name] = np.random.random(shape).astype(elem_type)
        # print(inp.name)
    return ret

def conv_params_from_onnx_node(graph, node):
    """ Create a Conv2DParams structure from an ONNX convolution node """
    if node.op_type != 'Conv':
        raise TypeError("Expecting type 'Conv', but got type:'%s'" (node.op_type,))
    attrs = dict( (x.name,x) for x in node.attribute )

    # Padding: fail if there are different padding for different dimensions
    pads = attrs['pads'].ints
    p = pads[0]
    if not all([p == x for x in pads[1:]]):
        raise NotImplementedError("pads: %s not supported" % (pads,))

    # filter size is a bit tricky.
    # One option might be kernel_shape, but this does not include the full
    # information (e.g., it can be 3x3 which does not include the number
    # of layers. Instead, we try to use the weights.

    # We assume that the input is the first and weight is the second input.
    # We will need to do something smarter if this breaks.
    inputs_name = node.input[0]
    weights_name = node.input[1]

    # Try to find the input in the inputs or value info part of the graph
    for vi in chain(graph.value_info, graph.input):
        if vi.name == inputs_name:
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
    # XXX: For now, we ignore inp.dims[-4] which is the batch size
    i = conv.Conv2DInParams(
        w = inp.type.tensor_type.shape.dim[-1].dim_value,
        h = inp.type.tensor_type.shape.dim[-2].dim_value,
        d = inp.type.tensor_type.shape.dim[-3].dim_value
    )

    conv_ps = conv.Conv2DParams(
        i = i,
        f = f,
        p = p,
        # TODO: deal with strading
        s = 1,
        p_out = 0
    )

    #print("%s" % (conv_ps,))

