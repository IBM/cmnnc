# Copyright (c) 2019-2020, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

from pprint import pprint
import dataclasses as dc
import typing
import itertools

import numpy as np
import islpy as isl

import pipeline as pl
from util import xparams, xdict
import conv
from op_info import OpInfo_CONV
from object_info import ObjectInfo

RD_a = pl.IslAccess.RD
WR_a = pl.IslAccess.WR


def test_mxv():
    """ Test a single MxV operation """
    params = xparams({'n': 128 })

    s_ops = [
        pl.OpInfo("MxV", [
            RD_a("{{ S[i] -> x[j] : i = 0 and 0 <= j < {n} }}".format(**params)),
            WR_a("{{ S[i] -> y[j] : i = 0 and 0 <= j < {n} }}".format(**params)),
        ])
    ]
    stage = pl.Stage(pl.StageInfo(s_ops))

    # Objects
    objs_info = {
        'x': ObjectInfo(shape=(params.n,)),
        'y': ObjectInfo(shape=(params.n,)),
    }

    # Initialize matrix, and create core configuration
    # np.random.seed(666)
    m_shape = params.eval("(n,n)")
    m = np.random.rand(*m_shape)
    cconf = pl.CoreConf(m)

    # Initalize pipeline
    pline = pl.Pipeline([stage], objs_info, execute_ops=True)
    x = pline.get_object("x")
    x[...] = np.random.rand(params.n)

    # Configure pipeline
    pline.configure([cconf])

    # Execute a single tick and compare results
    pline.tick()
    y = pline.get_object("y")
    assert np.array_equal(y, np.matmul(m, x))

def test_conv1d():
    """ Test a single 1D convolution """
    eg_vals = xparams({'n': 10, 'k': 3, 'p': 1})

    s1_ops = [
        pl.OpInfo("MxV", [
            RD_a("[n,k,p] -> { S1[o1] -> in[j] : 0 <= o1 < ((n - k + 2*p) + 1) and o1 <= j < o1 + k }"),
            WR_a("[n,k,p] -> { S1[o1] -> out[j] : 0 <= o1 < ((n - k + 2*p) + 1) and j = o1 }"),
        ]),
    ]
    stage1 = pl.Stage(pl.StageInfo(s1_ops), eg_vals)
    objs_info = {
        'in':  ObjectInfo(shape=(eg_vals.n,), padding=eg_vals.p),
        'out': ObjectInfo(shape=eg_vals.eval("(n-k+1,)"), padding=eg_vals.p),
    }
    pline = pl.Pipeline([stage1], objs_info, execute_ops=True)

    conv1_ps = conv.Conv1DParams(
        i = conv.Conv1DInParams(w=eg_vals["n"], d=1),
        f = conv.Conv1DFiltParams(w=eg_vals["k"], d=1, l=1),
        p = 1,
        s = 1,
        p_out = 0,
    )

    # Set filters
    filters1 = np.random.rand(*conv1_ps.get_filters_shape())
    filters1_m = filters1.reshape(conv1_ps.eval("(f.l, f.d*f.w)"))
    cconf = pl.CoreConf(filters1_m)

    # Set input
    image1 = np.random.rand(*conv1_ps.get_input_shape())
    image1 = np.pad(image1, conv1_ps.get_input_padding())
    inp = pline.get_object('in')
    inp[...] = image1

    pline.configure([cconf])
    for _ in range(conv1_ps.o.w):
        pline.tick()
    out = pline.get_object('out')

    # Verify results
    output_simple = conv.conv1d_simple(image1, filters1, conv1_ps)
    # NB: conv1d_simple considers the depth dimension while our access
    # relations above do not
    np.testing.assert_allclose(output_simple[0,:], out)

def test_conv1d_conv1d():
    # TODO: enable execute_ops = True, and compare results

    # A 1D-convolution with one layer (simplest case)
    #
    # For N=12, K=3, zero padding, the code looks simething like this:
    #
    # Stage s1:
    #     for o1 ← range(0, 10) {
    #         in2[o1,:] ← MXV(in1[o1:(o1 + 3),:])
    #     }
    # Stage s2:
    #     for o2 ← range(0, 8) {
    #         out2[o2,:] ← MXV(in2[o2:(o2 + 3),:])
    #     }
    #

    # Example values
    # N: in1 size
    # K: kernel size
    # P: padding
    eg_vals = xparams({'n': 10, 'k': 3, 'p': 1})

    s1_ops = [
        pl.OpInfo("MxV", [
            RD_a("[n,k,p] -> { S1[o1] -> in1[j] : 0 <= o1 < ((n - k + 2*p) + 1) and o1 <= j < o1 + k }"),
            WR_a("[n,k,p] -> { S1[o1] -> in2[j] : 0 <= o1 < ((n - k + 2*p) + 1) and j = o1 + p}"),
        ]),
    ]
    stage1 = pl.Stage(pl.StageInfo(s1_ops), eg_vals)


    s2_ops = [
        pl.OpInfo("MxV", [
            RD_a("[n,k,p] -> { S2[o2] -> in2[j] : 0 <= o2 < (n-k+2*p) and  o2 <= j < o2 + k }"),
        ]),
    ]
    stage2 = pl.Stage(pl.StageInfo(s2_ops), eg_vals)

    objs_info = {
        'in1': ObjectInfo(shape=(eg_vals.n,), padding=eg_vals.p),
        'in2': ObjectInfo(shape=(eg_vals.eval("n-k+2*p+1"),), padding=eg_vals.p),
    }
    pprint(objs_info)

    pline = pl.Pipeline([stage1, stage2], objs_info)

    for i in range(13):
        pline.tick()


def test_conv2d():
    conv1_ps = conv.Conv2DParams(
        i = conv.Conv2DInParams(w=32, h=32, d=3),
        f = conv.Conv2DFiltParams(w=3, h=3, d=3, l=16),
        p = 1,
        s = 1,
        p_out = 0)


    s1_ops = [
        OpInfo_CONV(conv1_ps, s_id="S1", vin_id="V1", vout_id="V2")
    ]
    stage1 = pl.Stage(pl.StageInfo(s1_ops))

    objs_info = {
        'V1': conv1_ps.get_input_objectinfo(),
        'V2': conv1_ps.get_output_objectinfo(),
    }

    p = pl.Pipeline([stage1], objs_info, execute_ops=True)

    # Set filters
    filters1 = np.random.rand(*conv1_ps.get_filters_shape())
    filters_m = filters1.reshape(conv1_ps.eval("(f.l, f.d*f.h*f.w)"))
    cconf = pl.CoreConf(filters_m)

    # Set input
    image1 = np.random.rand(*conv1_ps.get_input_shape())
    image1 = np.pad(image1, conv1_ps.get_input_padding())
    vals1 = p.get_object('V1')
    vals1[...] = image1

    # Configure pipeline
    p.configure([cconf])

    # Execute piepline
    for _ in range(conv1_ps.o.h*conv1_ps.o.w):
        p.tick()
    vals2 = p.get_object('V2')

    # Verify results
    output_simple = conv.conv2d_simple(image1, filters1, conv1_ps)
    output_mxv = conv.conv2d_mxv(image1, filters1, conv1_ps)
    np.testing.assert_allclose(output_simple, output_mxv)
    np.testing.assert_array_equal(output_mxv, vals2)

def test_conv2d_conv2d():
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

    s1_ops = [
        OpInfo_CONV(conv1_ps, s_id="S1", vin_id="V1", vout_id="V2"),
    ]
    stage1 = pl.Stage(pl.StageInfo(s1_ops))

    s2_ops = [
        OpInfo_CONV(conv2_ps, s_id="S2", vin_id="V2", vout_id="V3"),
    ]
    stage2 = pl.Stage(pl.StageInfo(s2_ops))

    objs_info = {
        'V1': conv1_ps.get_input_objectinfo(),
        'V2': conv2_ps.get_input_objectinfo(),
        'V3': conv2_ps.get_output_objectinfo(),
    }

    p = pl.Pipeline([stage1,stage2], objs_info, execute_ops=True)

    filters1 = np.random.rand(*conv1_ps.get_filters_shape())
    filters_m1 = filters1.reshape(conv1_ps.eval("(f.l, f.d*f.h*f.w)"))
    cconf1 = pl.CoreConf(filters_m1)

    filters2 = np.random.rand(*conv2_ps.get_filters_shape())
    filters_m2 = filters2.reshape(conv2_ps.eval("(f.l, f.d*f.h*f.w)"))
    cconf2 = pl.CoreConf(filters_m2)

    image = np.random.rand(*conv1_ps.get_input_shape())
    image = np.pad(image, conv1_ps.get_input_padding())

    p.configure([cconf1,cconf2])

    vals1 = p.get_object('V1')
    print("vals1.shape=%s image.shape=%s" % (vals1.shape,image.shape))
    pprint(objs_info)
    vals1[...] = image

    while True:
        iters = p.tick()
        print("*"*80)
        for (s,i) in iters.items():
            print("%s: %s" % (s, i))
        print("*"*80)
        # input()
        if iters['S2'] == (0, conv2_ps.o.h - 1, conv2_ps.o.w - 1):
            break

    vals3 = p.get_object('V3')
    pprint(vals3.shape)

    output1 = conv.conv2d_simple(image, filters1, conv1_ps)
    output1 = np.pad(output1, conv2_ps.get_input_padding())
    output2 = conv.conv2d_simple(output1, filters2, conv2_ps)
    np.testing.assert_allclose(output2, vals3)
    print("DONE!")

def get_params():
    params = xparams()
    # IN: input size (w/o padding)
    # F1: filter size
    # P1: padding
    params.update({ 'IN': 10, 'F1': 3, 'P1': 1, 'S1': 1})
    # O1: output 1 size
    params.compute("O1",  "(IN - F1 + 2*P1) // S1 + 1")
    params.compute("O2",  "O1")
    #
    params.update({'F2': 3, 'P2': 1, 'S2': 1})
    params.compute("O3",  "(O1 - F2 + 2*P2) // S2 + 1")
    params.compute("OUT",  "max(O2,O3)")
    return params

def test_residual_1d():
    #  CONV1D ---> CONV1D ---> ADD
    #          |           ^
    #          |           |
    #          +---------- +
    #
    # Stage S1:
    #  - MxV (CONV1D)
    #     - PARAMS: P1, F1
    #     - INPUT:  IN
    #     - OUTPUT: O1
    #
    # Stage S2:
    #  - MxV (CONV1D)
    #     - PARAMS: P2, F2
    #     - INPUT:  O1
    #     - OUTPUT: O3 (internal)
    #  - ADD:
    #     - INPUT: O1, O3 (internal)
    #     - OUTPUT: OUT
    #
    # cross-stage Objects:
    #  IN: WRITER: NONE,     READER: S1/MxV
    #  O1: WRITER: S1/MxV,   READER: S2/MxV
    # OUT: WRITER: S2/ADD,   READER: NONE
    #
    # Objects have a single writer and reader
    # Stages might read or write more than one objects

    params = get_params()

    s1_ops = [
        pl.OpInfo("MxV", [
            RD_a("{{ S1[s1] -> IN[i1] : 0 <= s1 < {O1} and s1 <= i1 < s1 + {F1} }}".format(**params)),
            WR_a("{{ S1[s1] -> O1[o1] : 0 <= s1 < {O1} and o1 = s1 + {P2} }}".format(**params)),
        ])
    ]

    s2_ops = [
        pl.OpInfo("MxV", [
            RD_a("{{ S2[s2] -> O1[o1] : 0 <= s2 < {O3} and s2 <= o1 < s2 + {F2}}}".format(**params)),
            WR_a("{{ S2[s2] -> O3[o3] : 0 <= s2 < {O3} and o3 = s2 }}".format(**params)),
        ]),
        pl.OpInfo("ADD", [
            RD_a("{{ S2[s2] -> O1[o1]   : 0 <= s2 < {O3} and o1  = s2 }}".format(**params)),
            RD_a("{{ S2[s2] -> O3[o3]   : 0 <= s2 < {O3} and o3  = s2 }}".format(**params)),
            WR_a("{{ S2[s2] -> OUT[out] : 0 <= s2 < {O3} and out = s2 }}".format(**params)),
        ])
    ]

    s2 = pl.Stage(pl.StageInfo(s2_ops))
    assert s2.si.ro_objs == set(('O1',))
    assert s2.si.wo_objs == set(('OUT',))
    assert s2.si.rw_objs == set(('O3',))

    s1 = pl.Stage(pl.StageInfo(s1_ops))
    assert s1.si.ro_objs == set(('IN',))
    assert s1.si.wo_objs == set(('O1',))
    assert s1.si.rw_objs == set()

    conv1_ps = conv.Conv1DParams(
        i = conv.Conv1DInParams(w=params.IN, d=1),
        f = conv.Conv1DFiltParams(w=params.F1, d=1, l=1),
        p = params.P1,
        s = params.S1,
        p_out = params.P2,
    )
    conv2_ps = conv.Conv1DParams(
        i = conv1_ps.o.to_in(),
        f = conv.Conv1DFiltParams(w=params.F2, d=1, l=1),
        p = params.P2,
        s = params.S2,
        p_out = 0,
    )

    objs_info = {
        # 'IN':  (params.eval("IN + 2*P1"), ),
        # 'O1':  (params.eval("O1 + 2*P2"), ),
        # 'O3':  (params.O3, ),
        # 'OUT': (params.OUT,),
        'IN': ObjectInfo(shape=(params.IN,), padding=params.P1),
        'O1': ObjectInfo(shape=(params.O1,), padding=params.P2),
        'O3': ObjectInfo(shape=(params.O3,), padding=0),
        'OUT': ObjectInfo(shape=(params.OUT,), padding=0),
    }
    pprint(objs_info)

    pline = pl.Pipeline([s1, s2], objs_info, execute_ops=True, loop_inp_limit=1)

    pprint(params)
    filters1 = np.random.rand(*conv1_ps.get_filters_shape())
    filters1_m = filters1.reshape(conv1_ps.eval("(f.l, f.d*f.w)"))
    cconf1 = pl.CoreConf(filters1_m)

    filters2 = np.random.rand(*conv2_ps.get_filters_shape())
    filters2_m = filters2.reshape(conv2_ps.eval("(f.l, f.d*f.w)"))
    cconf2 = pl.CoreConf(filters2_m)

    image = np.random.rand(*conv1_ps.get_input_shape())
    image = np.pad(image, conv1_ps.get_input_padding())
    inp = pline.get_object("IN")
    inp[...] = image

    pline.configure([cconf1, cconf2])

    print_info = False
    for iters in pline.tick_gen():
        if print_info:
            print("*"*80)
        for (s,i) in iters.items():
            if print_info:
                print("%s: %s" % (s, i))
        if print_info:
            print("*"*80)
    print("%s> DONE" % ("-"*30,))

    pline_out = pline.get_object('OUT')
    pline_o1 = pline.get_object('O1')
    pline_o3 = pline.get_object('O3')


    o1 = conv.conv1d_simple(image, filters1, conv1_ps)
    o2 = np.copy(o1)
    o1 = np.pad(o1, conv2_ps.get_input_padding())
    np.testing.assert_allclose(o1[0,:], pline_o1, err_msg="O1 does not match")
    o3 = conv.conv1d_simple(o1, filters2, conv2_ps)
    out = o3 + o2
    np.testing.assert_allclose(o3[0,:], pline_o3, err_msg="O3 does not match")
    np.testing.assert_allclose(out[0,:], pline_out, err_msg="OUT does not match")



if __name__ == '__main__':
    # test_mxv()
    # test_conv1d()
    # test_conv1d_conv1d()
    # test_conv2d()
    # test_conv2d_conv2d()
    # test_residual_1d()
    pass
