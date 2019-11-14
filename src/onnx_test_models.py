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

import onnxruntime as onnxrt
import unittest

def test_conv1():
    conv_ps = conv.ConvParams(
        i = conv.ConvInParams(w=32, h=32, d=3),
        f = conv.ConvFiltParams(w=3, h=3, d=3, l=16),
        p = 1,
        s = 1,
        p_out = 0
    )
    onnx_model = mk_conv_1(conv_ps)
    inp = onnx_rand_in(onnx_model)
    onnx.save(onnx_model, 'mymodel.onnx')
    sess = onnxrt.InferenceSession('mymodel.onnx')
    out = sess.run(None, inp)

    image  = np.pad(inp["conv1.in"][0], conv_ps.get_padding())
    filters = inp["conv1.ws"]
    exp_out = conv.conv2d_mxv(image, filters, conv_ps)
    np.testing.assert_allclose(out[0][0], exp_out, rtol=1e-06)

if __name__ == '__main__':
    # test_conv1()
