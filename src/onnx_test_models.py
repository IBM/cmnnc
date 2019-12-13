# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:


# Manually make some ONNX models using CONV for testing

from collections import namedtuple

import numpy as np
import onnx
import onnx.shape_inference

import conv

# quick-n-dirty tensor type
TensorTy = namedtuple('TensorTy',
                      ['shape', 'dty'],
                      defaults=(onnx.TensorProto.FLOAT,))

def mk_conv(conv_ps):
    """ CONV """

    if not isinstance(conv_ps, conv.Conv2DParams):
        raise TypeError("Expecting Conv2DParams, but got: %s" % (type(conv_ps),))

    # Tensor types
    tensor_tys = {
         # NB: [1] + is for the batch size
         'conv1.in' : TensorTy([1] + list(conv_ps.get_input_shape()) ),
         'conv1.ws' : TensorTy(shape=conv_ps.get_filters_shape()),
         'conv1.out': TensorTy([1] + list(conv_ps.get_output_shape())),
    }

    # Tensor value infos
    tensor_vis = {}
    for (name, tty) in tensor_tys.items():
        tensor_vis[name] = onnx.helper.make_tensor_value_info(name, tty.dty, tty.shape)

    # ONNX Node
    conv1 = onnx.helper.make_node(
        "Conv",                    # Name
        ["conv1.in",  "conv1.ws"], # Inputs
        ["conv1.out"],             # Outputs
        # Attribute
        pads = [conv_ps.p]*4,
    )

    # Tensor initlializer values
    tensor_init_vs = {}
    for n in ("conv1.ws", ):
        tty = tensor_tys[n]
        elem_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tty.dty]
        vals = np.random.random(tty.shape).astype(elem_type)
        tensor_init_vs[n] = onnx.helper.make_tensor(n, tty.dty, tty.shape, vals.flatten().tolist())


    # ONNX Graph
    graph_def = onnx.helper.make_graph(
        [conv1],
        'conv1-model',
        [tensor_vis['conv1.in']], # inputs
        [tensor_vis['conv1.out']], # outputs
        [tensor_init_vs['conv1.ws']], # initializer
    )

    model_def = onnx.helper.make_model(graph_def)
    model_def = onnx.shape_inference.infer_shapes(model_def, check_type=True)
    onnx.checker.check_model(model_def, full_check=True)
    return model_def


def mk_conv_conv(conv1_ps, conv2_ps):
    """ CONV -> CONV """

    if not isinstance(conv1_ps, conv.Conv2DParams):
        raise TypeError("Expecting Conv2DParams, but got: %s" % (type(conv1_ps),))
    if not isinstance(conv2_ps, conv.Conv2DParams):
        raise TypeError("Expecting Conv2DParams, but got: %s" % (type(conv2_ps),))

    # Tensor types
    tensor_tys = {
         # NB: [1] + is for the batch size
         'conv1.in' : TensorTy(shape=[1] + list(conv1_ps.get_input_shape()) ),
         'conv1.ws' : TensorTy(shape=conv1_ps.get_filters_shape()),
         'conv1.out': TensorTy(shape=[1] + list(conv1_ps.get_output_shape())),

         'conv2.in' : TensorTy(shape=[1] + list(conv2_ps.get_input_shape()) ),
         'conv2.ws' : TensorTy(shape=conv2_ps.get_filters_shape()),
         'conv2.out': TensorTy(shape=[1] + list(conv2_ps.get_output_shape())),
    }

    # Tensor values
    tensor_vs = {}
    for (name, tty) in tensor_tys.items():
        # print(name, tty)
        tensor_vs[name] = onnx.helper.make_tensor_value_info(name, tty.dty, tty.shape)

    # ONNX Nodes
    conv1 = onnx.helper.make_node(
        "Conv",                    # Name
        ["conv1.in",  "conv1.ws"], # Inputs
        ["conv1.out"],             # Outputs
        # Attributes
        pads = [conv1_ps.p]*4,
    )

    conv2 = onnx.helper.make_node(
        "Conv",                     # Name
        ["conv1.out",  "conv2.ws"], # Inputs
        ["conv2.out"],              # Outputs
        # Attributes
        pads = [conv2_ps.p]*4,
    )

    # Tensor initlializer values
    tensor_init_vs = {}
    for n in ("conv1.ws", "conv2.ws", ):
        tty = tensor_tys[n]
        elem_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tty.dty]
        vals = np.random.random(tty.shape).astype(elem_type)
        tensor_init_vs[n] = onnx.helper.make_tensor(n, tty.dty, tty.shape, vals.flatten().tolist())

    # ONNX Graph
    graph_def = onnx.helper.make_graph(
        [conv1, conv2],
        'conv2-model',
        [tensor_vs['conv1.in'], ],   # inputs
        [tensor_vs['conv2.out']],    # outputs
        [tensor_init_vs['conv1.ws'], # Initializers
         tensor_init_vs['conv2.ws']],
    )

    model_def = onnx.helper.make_model(graph_def)
    model_def = onnx.shape_inference.infer_shapes(model_def, check_type=True)
    onnx.checker.check_model(model_def, full_check=True)
    return model_def

#  CONV2D ---> CONV2D ---> ADD
#          |                ^
#          |                |
#          +--------------- +
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

def mk_simple_residual(conv1_ps: conv.Conv2DParams, conv2_ps: conv.Conv2DParams):
    if not isinstance(conv2_ps, conv.Conv2DParams):
        raise TypeError("Expecting Conv2DParams, but got: %s" % (type(conv1_ps),))
    if not isinstance(conv2_ps, conv.Conv2DParams):
        raise TypeError("Expecting Conv2DParams, but got: %s" % (type(conv2_ps),))

    # Tensor types
    tensor_tys = {
         # NB: "[1] +" adds a batch size, which is needed for the convolution operator
        'in' : TensorTy(shape=[1] + list(conv1_ps.get_input_shape())),
        'w1' : TensorTy(shape=conv1_ps.get_filters_shape()),
        'v1': TensorTy(shape=[1] + list(conv1_ps.get_output_shape())),

        'w2' : TensorTy(shape=conv2_ps.get_filters_shape()),
        'v2': TensorTy(shape=[1] + list(conv2_ps.get_input_shape())),
    }
    assert tensor_tys['v1'].shape == tensor_tys['v2'].shape
    tensor_tys['out'] = TensorTy(shape=tensor_tys['v2'].shape)

    # Tensor values
    tensor_vs = {}
    for (name, tty) in tensor_tys.items():
        tensor_vs[name] = onnx.helper.make_tensor_value_info(name, tty.dty, tty.shape)

    # ONNX Nodes
    conv1 = onnx.helper.make_node(
        "Conv",        # Name
        ["in",  "w1"], # Inputs
        ["v1"],  # Outputs
        # Attributes
        pads = [conv1_ps.p]*4,
        name = "Conv1",
    )

    conv2 = onnx.helper.make_node(
        "Conv",        # Name
        ["v1",  "w2"], # Inputs
        ["v2"],        # Outputs
        # Attributes
        pads = [conv2_ps.p]*4,
        name = "Conv2",
    )

    add = onnx.helper.make_node(
        "Add",         # Name
        ["v1",  "v2"], # Inputs
        ["out"],       # Outputs
        # Attributes
        name = "Add2",
    )

    # Tensor initlializer values
    tensor_init_vs = {}
    for n in ("w1", "w2", ):
        tty = tensor_tys[n]
        elem_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tty.dty]
        vals = np.random.random(tty.shape).astype(elem_type)
        tensor_init_vs[n] = onnx.helper.make_tensor(n, tty.dty, tty.shape, vals.flatten().tolist())

    # ONNX Graph
    graph_def = onnx.helper.make_graph(
        [conv1, conv2, add],
        'simple-residual-model',
        [tensor_vs['in'], ],   # inputs
        [tensor_vs['out']],    # outputs
        [tensor_init_vs['w1'], # Initializers
         tensor_init_vs['w2']],
    )

    model_def = onnx.helper.make_model(graph_def)
    model_def = onnx.shape_inference.infer_shapes(model_def, check_type=True)
    onnx.checker.check_model(model_def, full_check=True)
    return model_def
