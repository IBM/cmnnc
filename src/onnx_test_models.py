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

    if not isinstance(conv_ps, conv.ConvParams):
        raise TypeError("Expecting ConvParams, but got: %s" % (type(conv_ps),))

    # Tensor types
    tensors_tys = {
         # NB: [1] + is for the batch size
         'conv1.in' : TensorTy([1] + list(conv_ps.get_image_shape()) ),
         'conv1.ws' : TensorTy(shape=conv_ps.get_filters_shape()),
         'conv1.out': TensorTy([1] + list(conv_ps.get_out_shape())),
    }

    # Tensor values
    tensor_vs = {}
    for (name, tty) in tensors_tys.items():
        tensor_vs[name] = onnx.helper.make_tensor_value_info(name, tty.dty, tty.shape)

    # ONNX Node
    conv1 = onnx.helper.make_node(
        "Conv",                    # Name
        ["conv1.in",  "conv1.ws"], # Inputs
        ["conv1.out"],             # Outputs
        # Attribute
        pads = [conv_ps.p]*4,
    )


    # ONNX Graph
    graph_def = onnx.helper.make_graph(
        [conv1],
        'conv1-model',
        [tensor_vs['conv1.in'], tensor_vs['conv1.ws']], # inputs
        [tensor_vs['conv1.out']], # outputs
        [], # initializer
    )

    model_def = onnx.helper.make_model(graph_def)
    model_def = onnx.shape_inference.infer_shapes(model_def, check_type=True)
    onnx.checker.check_model(model_def, full_check=True)
    return model_def


def mk_conv_conv(conv1_ps, conv2_ps):
    """ CONV -> CONV """

    if not isinstance(conv1_ps, conv.ConvParams):
        raise TypeError("Expecting ConvParams, but got: %s" % (type(conv1_ps),))
    if not isinstance(conv1_ps, conv.ConvParams):
        raise TypeError("Expecting ConvParams, but got: %s" % (type(conv1_ps),))

    # Tensor types
    tensors_tys = {
         # NB: [1] + is for the batch size
         'conv1.in' : TensorTy(shape=[1] + list(conv1_ps.get_image_shape()) ),
         'conv1.ws' : TensorTy(shape=conv1_ps.get_filters_shape()),
         'conv1.out': TensorTy(shape=[1] + list(conv1_ps.get_out_shape())),

         'conv2.in' : TensorTy(shape=[1] + list(conv2_ps.get_image_shape()) ),
         'conv2.ws' : TensorTy(shape=conv2_ps.get_filters_shape()),
         'conv2.out': TensorTy(shape=[1] + list(conv2_ps.get_out_shape())),
    }

    # Tensor values
    tensor_vs = {}
    for (name, tty) in tensors_tys.items():
        # print(name, tty)
        tensor_vs[name] = onnx.helper.make_tensor_value_info(name, tty.dty, tty.shape)

    # ONNX Nodes
    conv1 = onnx.helper.make_node(
        "Conv",                    # Name
        ["conv1.in",  "conv1.ws"], # Inputs
        ["conv1.out"],             # Outputs
        # Attribute
        pads = [conv1_ps.p]*4,
    )

    conv2 = onnx.helper.make_node(
        "Conv",                     # Name
        ["conv1.out",  "conv2.ws"], # Inputs
        ["conv2.out"],              # Outputs
        # Attribute
        pads = [conv2_ps.p]*4,
    )


    # ONNX Graph
    graph_def = onnx.helper.make_graph(
        [conv1, conv2],
        'conv2-model',
        [tensor_vs['conv1.in'], tensor_vs['conv1.ws'], tensor_vs['conv2.ws']], # inputs
        [tensor_vs['conv2.out']], # outputs
        [],
    )

    model_def = onnx.helper.make_model(graph_def)
    model_def = onnx.shape_inference.infer_shapes(model_def, check_type=True)
    onnx.checker.check_model(model_def, full_check=True)
    return model_def
