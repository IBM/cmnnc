# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import numpy as np
import onnx
import onnx.mapping
import onnxruntime as onnxrt

from onnx_test_models import mk_conv_1 as onnx_mk_conv_1

def test_true():
    assert True

def onnx_rand_in(model):
    """ Create random inputs for a given ONNX model """
    ret = {}
    for inp in model.graph.input:
        tensor_ty = inp.type.tensor_type
        elem_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_ty.elem_type]
        shape = [d.dim_value for d in tensor_ty.shape.dim]
        ret[inp.name] = np.random.random(shape).astype(elem_type)
    return ret

def test_conv1():
    onnx_model = onnx_mk_conv_1()
    onnx_in = onnx_rand_in(onnx_model)
    onnx.save(onnx_model, 'mymodel.onnx')
    sess = onnxrt.InferenceSession('mymodel.onnx')
    # conv1_in = np.random.random(tensors['conv1.in'].shape).astype(np.float32)
    # conv1_ws = np.random.random(tensors['conv1.ws'].shape).astype(np.float32)
    out = sess.run(None, {'conv1.in': conv1_in, 'conv1.ws': conv1_ws})

