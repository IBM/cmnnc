# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:


# Manually make some conv models inspired by (cifar) resnet-20 for testing.

from collections import namedtuple

import numpy as np
import onnx
import onnx.shape_inference

import conv

# quick-n-dirty tensor type
TensorTy = namedtuple('TensorTy',
                      ['shape', 'dty'],
                      defaults=(onnx.TensorProto.FLOAT,))

def mk_conv_1(conv_ps):
    """ CONV """

    if not isinstance(conv_ps, conv.ConvParams):
        raise TypeError("Expecting ConvParams, but got: %s" % (type(conv_ps),))

    # Tensor types
    tensors_tys = {
         # NB: [1] + is for the batch size
         'conv1.in' : TensorTy(shape=[1] + list(conv_ps.get_image_shape()) ),
         'conv1.ws' : TensorTy(shape=conv_ps.get_filters_shape()),
         'conv1.out': TensorTy(shape=[1] + list(conv_ps.get_out_shape())),
    }

    # Tensor values
    tensor_vs = {}
    for (name, tty) in tensors_tys.items():
        tensor_vs[name] = onnx.helper.make_tensor_value_info(name, tty.dty, tty.shape)

    attrs = {
        "pads": onnx.helper.make_attribute("pads", [conv_ps.p, conv_ps.p, conv_ps.p, conv_ps.p]),
    }

    # ONNX Node
    conv1 = onnx.helper.make_node(
        "Conv",                    # Name
        ["conv1.in",  "conv1.ws"], # Inputs
        ["conv1.out"],             # Outputs
        # Attribute
        pads = [conv_ps.p, conv_ps.p, conv_ps.p, conv_ps.p],
    )


    # ONNX Graph
    graph_def = onnx.helper.make_graph(
        [conv1],
        'conv1-model',
        [tensor_vs['conv1.in'], tensor_vs['conv1.ws']], # inputs
        [tensor_vs['conv1.out']], # outputs
    )

    model_def = onnx.helper.make_model(graph_def)
    onnx.checker.check_model(model_def)

    model_def = onnx.shape_inference.infer_shapes(model_def)
    onnx.checker.check_model(model_def)
    return model_def

