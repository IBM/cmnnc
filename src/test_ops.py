# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import unittest
import numpy as np
import onnx
import onnxruntime as onnxrt

import conv
from onnx_test_models import mk_conv_1 as onnx_mk_conv_1

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

class TestConv(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # same parameters as CIFAR's first convolution
        conv_ps = self.conv_ps = conv.ConvParams(
            i = conv.ConvInParams(w=32, h=32, d=3),
            f = conv.ConvFiltParams(w=3, h=3, d=3, l=16),
            p = 1,
            s = 1,
            p_out = 0
        )

    def test_mxv(self):
        """ Test MxV and simple versions of CONV 2D """
        # Initalize random filters and image
        conv_ps = self.conv_ps
        filters = np.random.rand(*conv_ps.get_filters_shape())
        image   = np.random.rand(*conv_ps.get_image_shape())
        image   = np.pad(image, conv_ps.get_padding())

        output_simple = conv.conv2d_simple(image, filters, conv_ps)
        output_mxv = conv.conv2d_mxv(image, filters, conv_ps)
        np.testing.assert_allclose(output_simple, output_mxv)

    def test_onnxrt(self):
        """ Test ONNX and simple versions of CONV 2D """
        onnx_model = onnx_mk_conv_1(self.conv_ps)
        inp = onnx_rand_in(onnx_model)
        onnx.save(onnx_model, 'mymodel.onnx')
        sess = onnxrt.InferenceSession('mymodel.onnx')
        out = sess.run(None, inp)
        image  = np.pad(inp["conv1.in"][0], self.conv_ps.get_padding())
        filters = inp["conv1.ws"]
        exp_out = conv.conv2d_mxv(image, filters, self.conv_ps)
        np.testing.assert_allclose(out[0][0], exp_out, rtol=1e-06)
