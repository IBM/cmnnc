# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import numpy as np

import pipeline as pl


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
