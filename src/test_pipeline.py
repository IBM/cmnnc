# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

from pprint import pprint
import dataclasses as dc
import typing

import numpy as np
import islpy as isl

import pipeline as pl
import conv


def test_mxv():
    params = {'n': 128 }

    # read access relation (vector x)
    rd_a = "{{ S[i] -> x[j] : i = 0 and 0 <= j < {n} }}".format(**params)
    # write access relation (vector y)
    wr_a = "{{ S[i] -> y[j] : i = 0 and 0 <= j < {n} }}".format(**params)
    # Define a stage based on the above relations
    stage = pl.Stage(pl.StageInfo(
        rd_a = rd_a,
        wr_a = wr_a,
    ))

    # Objects
    objs = {
        'x': (params['n'], ),
        'y': (params['n'], )
    }

    # Initialize matrix, and create core configuration
    # np.random.seed(666)
    m_shape = eval("(n,n)", params)
    m = np.random.rand(*m_shape)
    cconf = pl.CoreConf(m)

    # Initalize pipeline
    pline = pl.Pipeline([stage], objs, execute_ops=True)
    x = pline.get_object("x")
    x[...] = np.random.rand(params['n'])

    # Configure pipeline
    pline.configure([cconf])

    # Execute a single tick and compare results
    pline.tick()
    y = pline.get_object("y")
    assert np.array_equal(y, np.matmul(m, x))

def test_conv1d():
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
    eg_vals = {'n': 10, 'k': 3, 'p': 1}

    stage1 = pl.Stage(pl.StageInfo(
        rd_a = "[n,k,p] -> { S1[o1] -> in1[j] : 0 <= o1 < ((n - k + 2*p) + 1) and o1 <= j < o1 + k }",
        wr_a = "[n,k,p] -> { S1[o1] -> in2[j] : 0 <= o1 < ((n - k + 2*p) + 1) and j = o1 + p}"
    ), eg_vals)


    stage2 = pl.Stage(pl.StageInfo(
        rd_a = "[n,k,p] -> { S2[o2] -> in2[j] : 0 <= o2 < (n-k+2*p) and  o2 <= j < o2 + k }"
    ), eg_vals)

    objects = {
        'in1': eval("(n + 2*p,)", eg_vals),
        'in2': eval("(n - k + 2*p + 1 + 2*p,)", eg_vals),
    }
    pprint(objects)

    pline = pl.Pipeline([stage1, stage2], objects)

    for i in range(13):
        pline.tick()


def test_conv2d():
    conv1_ps = conv.ConvParams(
        i = conv.ConvInParams(w=32, h=32, d=3),
        f = conv.ConvFiltParams(w=3, h=3, d=3, l=16),
        p = 1,
        s = 1,
        p_out = 0)

    s1_rdwr_a = conv1_ps.get_rd_wr_a(s_id=1, vin_id=1, vout_id=2)
    stage1 = pl.Stage(pl.StageInfo(
        rd_a = s1_rdwr_a[0],
        wr_a = s1_rdwr_a[1],
    ))

    objs = {
        'V1': conv1_ps.get_in_shape(),
        'V2': conv1_ps.get_out_shape(),
    }

    p = pl.Pipeline([stage1], objs, execute_ops=True)

    # Set filters
    filters1 = np.random.rand(*conv1_ps.get_filters_shape())
    filters_m = filters1.reshape(conv1_ps.eval("(f.l, f.d*f.h*f.w)"))
    cconf = pl.CoreConf(filters_m)

    # Set input
    image1 = np.random.rand(*conv1_ps.get_image_shape())
    image1 = np.pad(image1, conv1_ps.get_padding())
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

    conv1_ps = conv.ConvParams(
        i = conv.ConvInParams(w=32, h=32, d=3),
        f = conv.ConvFiltParams(w=3, h=3, d=3, l=1),
        p = conv1_padding,
        p_out = conv2_padding,
        s = 1)

    conv2_ps = conv.ConvParams(
        i = conv1_ps.o.to_in(),
        f = conv.ConvFiltParams(w=3, h=3, d=conv1_ps.f.l, l=1),
        p = conv2_padding,
        p_out = 0,
        s = 1)

    s1_rdwr_a = conv1_ps.get_rd_wr_a(s_id=1, vin_id=1, vout_id=2)
    print("S1\n R:\n%s\n W:\n%s\n" % s1_rdwr_a)
    s2_rdwr_a = conv2_ps.get_rd_wr_a(s_id=2, vin_id=2, vout_id=3)

    stage1 = pl.Stage(pl.StageInfo(
        rd_a = s1_rdwr_a[0],
        wr_a = s1_rdwr_a[1],
    ))

    stage2 = pl.Stage(pl.StageInfo(
         rd_a = s2_rdwr_a[0],
         wr_a = s2_rdwr_a[1],
    ))

    objs = {
        'V1': conv1_ps.get_in_shape(),
        'V2': conv2_ps.get_in_shape(),
        'V3': conv2_ps.get_out_shape(),
    }

    p = pl.Pipeline([stage1,stage2], objs, execute_ops=True)

    filters1 = np.random.rand(*conv1_ps.get_filters_shape())
    filters_m1 = filters1.reshape(conv1_ps.eval("(f.l, f.d*f.h*f.w)"))
    cconf1 = pl.CoreConf(filters_m1)

    filters2 = np.random.rand(*conv2_ps.get_filters_shape())
    filters_m2 = filters2.reshape(conv2_ps.eval("(f.l, f.d*f.h*f.w)"))
    cconf2 = pl.CoreConf(filters_m2)

    image = np.random.rand(*conv1_ps.get_image_shape())
    image = np.pad(image, conv1_ps.get_padding())

    p.configure([cconf1,cconf2])

    vals1 = p.get_object('V1')
    print("vals1.shape=%s image.shape=%s" % (vals1.shape,image.shape))
    stage2.print_loc_to_max_iter()
    pprint(objs)
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
    output1 = np.pad(output1, conv2_ps.get_padding())
    output2 = conv.conv2d_simple(output1, filters2, conv2_ps)
    np.testing.assert_allclose(output2, vals3)
    print("DONE!")

class xdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def str_to_isl_map(x: str) -> isl.Map:
    try:
        return isl.Map(x)
    except:
        print("Failed to create an isl.Map from %s" % (x,))
        raise

@dc.dataclass(init=False)
class OpInfo:
    """ Operation (polyhedral) info.

    A list of Instace space -> object access, per object for reads and writes
    """
    oi_rds: typing.List[isl.Map]
    oi_wrs: typing.List[isl.Map]

    def __init__(self, op_name: str):
        self.oi_rds = []
        self.oi_wrs = []
        self.op_name = op_name

    def add_rd_a(self, rel):
        """ Add a read access relation for MxV """
        if isinstance(rel, str):
            rel = str_to_isl_map(rel)
        self.oi_rds.append(rel)
        return self

    def add_wr_a(self, rel):
        """ Add a write access relation for MxV """
        if isinstance(rel, str):
            rel = str_to_isl_map(rel)
        self.oi_wrs.append(rel)
        return self

@dc.dataclass(init=False)
class StageInfo:
    """ Polyhedral information for a stage

    The idea here is that we are able to express the code for every stage in a
    fused loop as follows:
      for i in ...
          for j in ...
              ....
                  MxV()
                  DPU_INS1()
                  DPU_INS2()
                  ....

    Each operation (MXV, DPU_INS) has a bunch of read/write accesses to objects
    that are represented by the polyhedral information.
    """
    mxv_i: OpInfo
    dpu_i: typing.List[OpInfo]

    def __init__(self):
        self.mxv_i = OpInfo("MxV")
        self.dpu_i = []

def test_residual_1d():
    #  CONV1D ---> CONV1D ---> ADD
    #          |           ^
    #          |           |
    #          +---------- +
    #
    #
    # Stage S1:
    #  - MxV (CONV1D)
    #     - PARAMS: P1, F1
    #     - INPUT:  IN
    #     - OUTPUT: O1, O2
    #
    # Stage S2:
    #  - MxV (CONV1D)
    #     - PARAMS: P2, F2
    #     - INPUT:  O1
    #     - OUTPUT: O3 (internal)
    #  - ADD:
    #     - INPUT: O2, O3 (internal)
    #     - OUTPUT: OUT
    #
    params = {}
    def params_compute(p, expr):
        params[p] = eval(expr, None, params)

    # IN: input size (w/o padding)
    # F1: filter size
    # P1: padding
    params = { 'IN': 10, 'F1': 3, 'P1': 1, 'S1': 1}
    # O1: output 1 size
    params_compute("O1",  "(IN - F1 + 2*P1) // S1 + 1")
    params_compute("O2",  "O1")
    #
    params.update({'F2': 3, 'P2': 1, 'S2': 1})
    params_compute("O3",  "(O1 - F2 + 2*P2) // S2 + 1")

    s1 = StageInfo()
    s1.mxv_i.add_rd_a("{{ S1[s1] -> IN[i1]: 0 <= s1 <= {O1} and  s1 <= i1 < s1 + {F1} }}".format(**params))
    s1.mxv_i.add_wr_a("{{ S1[s1] -> O1[o1] : o1 = s1 + {P2} }}".format(**params))
    s1.mxv_i.add_wr_a("{{ S1[s1] -> O2[o2] : o2 = s1 }}".format(**params))


    # NB: we do not track the writes and reads on O3, because it's local (within a single stage)
    s2 = StageInfo()
    s2.mxv_i.add_rd_a("{{ S2[s2] -> O1[o1] : 0 <= s2 <= {O3} and s2 <= o1 < s2 + {F2}}}".format(**params))
    add_info = OpInfo("ADD")
    add_info.add_rd_a("{{ S2[s2] -> O2[o2] : o2 = s2 }}".format(**params))
    add_info.add_wr_a("{{ S2[s2] -> OUT[out] : out = s2 }}".format(**params))
    s2.dpu_i.append(add_info)

    return


if __name__ == '__main__':
    a = test_residual_1d()
