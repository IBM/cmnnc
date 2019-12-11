# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import onnx
from onnx_util import  conv_params_from_onnx_node

# if we want to incrementally build and test the graph, this might help: 
# https://github.com/microsoft/onnxruntime/issues/1455

def test_resnet():
    m = onnx.load("onnx/resnet20-cifar.onnx")
    graph = m.graph
    conv = graph.node[0]
    params = conv_params_from_onnx_node(graph, conv)
    return m

if __name__ == '__main__':
    ret = test_resnet()
