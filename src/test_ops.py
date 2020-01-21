# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import unittest
import numpy as np
import onnx
import onnx.numpy_helper
import onnxruntime as onnxrt

import conv
from onnx_test_models import mk_conv as onnx_mk_conv
from onnx_test_models import mk_conv_conv as onnx_mk_conv_conv
from onnx_util import onnx_get_init_data, onnx_rand_input


class TestConv(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # same parameters as CIFAR's first convolution
        self.conv_ps = conv.Conv2DParams(
            i=conv.Conv2DInParams(w=32, h=32, d=3),
            f=conv.Conv2DFiltParams(w=3, h=3, d=3, l=16),
            p=1,
            s=1,
            p_out=0,
        )

    def test_mxv(self):
        """ Test MxV and simple versions of CONV 2D """
        # Initalize random filters and image
        conv_ps = self.conv_ps
        filters = np.random.rand(*conv_ps.get_filters_shape())
        image = np.random.rand(*conv_ps.get_input_shape())
        image = np.pad(image, conv_ps.get_input_padding())

        output_simple = conv.conv2d_simple(image, filters, conv_ps)
        output_mxv = conv.conv2d_mxv(image, filters, conv_ps)
        np.testing.assert_allclose(output_simple, output_mxv)

    def test_onnxrt(self):
        """ Test ONNX and simple versions of CONV 2D """
        onnx_model = onnx_mk_conv(self.conv_ps)
        inp = onnx_rand_input(onnx_model)
        onnx.save(onnx_model, "mymodel.onnx")
        sess = onnxrt.InferenceSession("mymodel.onnx")
        out = sess.run(None, inp)
        image = np.pad(inp["conv1.in"][0], self.conv_ps.get_input_padding())
        filters = onnx.numpy_helper.to_array(
            onnx_get_init_data(onnx_model.graph, "conv1.ws")
        )
        exp_out = conv.conv2d_mxv(image, filters, self.conv_ps)
        np.testing.assert_allclose(out[0][0], exp_out, rtol=1e-06)


class TestConvConv(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # same parameters as CIFAR's convolutions
        conv1_padding = 1
        conv2_padding = 1

        conv1_ps = self.conv1_ps = conv.Conv2DParams(
            i=conv.Conv2DInParams(w=32, h=32, d=3),
            f=conv.Conv2DFiltParams(w=3, h=3, d=3, l=1),
            p=conv1_padding,
            p_out=conv2_padding,
            s=1,
        )

        self.conv2_ps = conv.Conv2DParams(
            i=conv1_ps.o.to_in(),
            f=conv.Conv2DFiltParams(w=3, h=3, d=conv1_ps.f.l, l=1),
            p=conv2_padding,
            p_out=0,
            s=1,
        )

    def test_onnxrt(self):
        """ Test ONNX and simple versions of CONV 2D """
        onnx_model = onnx_mk_conv_conv(self.conv1_ps, self.conv2_ps)
        inp = onnx_rand_input(onnx_model)
        onnx.save(onnx_model, "mymodel.onnx")
        sess = onnxrt.InferenceSession("mymodel.onnx")
        out = sess.run(None, inp)

        image = np.pad(inp["conv1.in"][0], self.conv1_ps.get_input_padding())
        filters1 = onnx.numpy_helper.to_array(
            onnx_get_init_data(onnx_model.graph, "conv1.ws")
        )
        filters2 = onnx.numpy_helper.to_array(
            onnx_get_init_data(onnx_model.graph, "conv2.ws")
        )

        exp_out1 = conv.conv2d_mxv(image, filters1, self.conv1_ps)
        exp_out1 = np.pad(exp_out1, self.conv2_ps.get_input_padding())
        exp_out2 = conv.conv2d_mxv(exp_out1, filters2, self.conv2_ps)

        np.testing.assert_allclose(out[0][0], exp_out2, rtol=1e-06)
