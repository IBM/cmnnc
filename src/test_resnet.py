# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import onnx
from onnx_util import  onnx_conv_get_params

# if we want to incrementally build and test the graph, this might help: 
# https://github.com/microsoft/onnxruntime/issues/1455

def test_resnet():
    m = onnx.load("onnx/resnet20-cifar.onnx")
    graph = m.graph
    conv = graph.node[0]
    params = onnx_conv_get_params(graph, conv)
    return m

if __name__ == '__main__':
    ret = test_resnet()
