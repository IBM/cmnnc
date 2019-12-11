# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import typing

import onnxruntime as onnxrt
import onnx

import conv
from pipeline import StageInfo
from onnx_test_models import mk_simple_residual as onnx_mk_simple_residual
from onnx_util import onnx_rand_in


# TODO: move this to another file when done
def onnx_partition(onnx_model) -> typing.List[StageInfo]:
    pass

def test_onnx_residual_2d():
    # Create the following ONNX graph:
    #
    #  CONV2D ---> CONV2D ---> ADD
    #          |                ^
    #          |                |
    #          +--------------- +
    #
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
    inp = onnx_rand_in(onnx_m)

    # Parse onnx graph to create a pipeline

    # Configure the pipeline based using the ONNX initializers

    # Execute the pipeline
    # image  = np.pad(inp["in"][0], self.conv_ps.get_padding())

    # Execute using onnxruntime
    onnx.save(onnx_m, 'simple_residual_2d.onnx')
    sess = onnxrt.InferenceSession('simple_residual_2d.onnx')
    out = sess.run(None, inp)

if __name__ == '__main__':
    test_onnx_residual_2d()
