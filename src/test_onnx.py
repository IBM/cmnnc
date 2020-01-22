# Copyright (c) 2019-2020, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import typing
import dataclasses as dc
from pprint import pprint

import numpy as np
import onnxruntime as onnxrt
import onnx

import conv
import pipeline as pl
from onnx_test_models import mk_simple_residual as onnx_mk_simple_residual
from onnx_util import onnx_rand_input
from onnx_graph import OnnxGraph


def test_onnx_residual_2d():
    # Create the following ONNX graph
    # (this is what onnx_mk_simple_residual does)
    #
    #  CONV2D ---> CONV2D ---> ADD
    #          |                ^
    #          |                |
    #          +--------------- +
    #
    # CONV2D
    #   input:  in
    #   output: v1
    #   weights: w1
    # CONV2D
    #   input:  v1
    #   output: v2
    #   weights: w2
    # ADD
    #  input: v1,v2
    #  output: out
    conv1_padding = 1
    conv2_padding = 1

    conv1_ps = conv.Conv2DParams(
        i=conv.Conv2DInParams(w=32, h=32, d=3),
        f=conv.Conv2DFiltParams(w=3, h=3, d=3, l=1),
        p=conv1_padding,
        p_out=conv2_padding,
        s=1,
    )

    conv2_ps = conv.Conv2DParams(
        i=conv1_ps.o.to_in(),
        f=conv.Conv2DFiltParams(w=3, h=3, d=conv1_ps.f.l, l=1),
        p=conv2_padding,
        p_out=0,
        s=1,
    )

    # create simple model with residual path
    onnx_m = onnx_mk_simple_residual(conv1_ps, conv2_ps)

    # create random input
    inp = onnx_rand_input(onnx_m)

    # Execute using onnxruntime
    onnx.save(onnx_m, "simple_residual_2d.onnx")
    sess = onnxrt.InferenceSession("simple_residual_2d.onnx")
    out = sess.run(None, inp)

    # Parse onnx graph, and create a pipeline
    graph = OnnxGraph(onnx_m)
    pprint(graph.partitions)
    pline = graph.get_pipeline()

    # set inputs
    for (inp_name, inp_data) in inp.items():
        obj_info = graph.objs_info[inp_name]
        assert inp_data.shape == (1,) + obj_info.shape  # NB: batching
        # data = np.random.rand(*obj_info.shape)
        data = inp_data[0]
        data = np.pad(data, obj_info.padding)
        obj = pline.get_object(inp_name)
        obj[...] = data


    # Execute the pipeline
    print_info = False
    for iters in pline.tick_gen():
        if print_info:
            print("*" * 80)
        for (s, i) in iters.items():
            if print_info:
                print("%s: %s" % (s, i))
        if print_info:
            print("*" * 80)
    print("%s> DONE" % ("-" * 30,))

    # Get pipeline results
    pline_out = pline.get_object("out")
    pline_v1 = pline.get_object("v1")
    pline_v2 = pline.get_object("v2")

    # Execute using manual ops
    in_m = np.pad(inp["in"][0], graph.objs_info["in"].padding)
    w1_m = np.array(graph.init_tvs["w1"].float_data).reshape(
        conv1_ps.get_filters_shape()
    )
    v1_m = conv.conv2d_simple(in_m, w1_m, conv1_ps)
    v1_m = np.pad(v1_m, graph.objs_info["v1"].padding)
    np.testing.assert_allclose(
        v1_m, pline_v1, err_msg="pipeline v1 does not match manual v1"
    )

    w2_m = np.array(graph.init_tvs["w2"].float_data).reshape(
        conv2_ps.get_filters_shape()
    )
    v2_m = conv.conv2d_simple(v1_m, w2_m, conv2_ps)
    v2_m = np.pad(v2_m, graph.objs_info["v2"].padding)
    np.testing.assert_allclose(
        v2_m, pline_v2, err_msg="pipeline v2 does not match manual v2"
    )

    np.testing.assert_allclose(
        out[0][0, :], pline_out, err_msg="OUT does not match", rtol=1e-06
    )

    return graph


if __name__ == "__main__":
    ret = test_onnx_residual_2d()
