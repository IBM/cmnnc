# Copyright (c) 2020, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import numpy as np

from object_info import ObjectInfo


def test_unpad():
    oi = ObjectInfo(shape=(1, 10, 10), padding=((0, 0), (1, 1), (1, 1)))
    a = np.random.random(oi.shape)
    a_padded = np.pad(a, oi.padding)
    a_slice = oi.get_unpadded_slice(a_padded)
    np.testing.assert_array_equal(a_slice, a)
