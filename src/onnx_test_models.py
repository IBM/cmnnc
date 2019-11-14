# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:


# Manually make some conv models inspired by (cifar) resnet-20 for testing.

from collections import namedtuple

import onnx
import onnx.shape_inference

# quick-n-dirty tensor type
TensorTy = namedtuple('TensorTy',
                      ['shape', 'dty'],
                      defaults=(onnx.TensorProto.FLOAT,))

def mk_conv_1():
    """ CONV """

    # Tensor types
    tensors_tys = {
        'conv1.in'  : TensorTy(shape=[1,3,32,32]),
        'conv1.ws'  : TensorTy(shape=[16,3,3,3]),
        'conv1.out' : TensorTy(shape=[1,16,30,30]),
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
