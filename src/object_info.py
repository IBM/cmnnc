# Copyright (c) 2020, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import dataclasses as dc
import typing

from util import check_class_hints

@dc.dataclass(init = False)
class ObjectInfo:
    shape:   typing.Tuple[int,...] # (unpadded) shape
    # NB: padding is something that can be passed to np.pad
    padding: typing.Optional[typing.Tuple[typing.Tuple[int,int], ...]]

    def __init__(self, shape, padding = None):
        self.shape = shape
        if padding is None:
           padding = tuple((0,0) for _ in self.shape)
        elif isinstance(padding, int):
           padding = tuple((padding,padding) for _ in self.shape)

        self.padding = padding
        assert len(self.padding) == len(self.shape)
        check_class_hints(self)

    def get_padded_shape(self):
        return tuple(x + p_s + p_e for (x, (p_s, p_e)) in zip(self.shape, self.padding))

    def get_unpadded_slice(self, np_arr):
        """ Given whatever padding this object has, return a non-padded slice """
        assert np_arr.shape == self.get_padded_shape()
        slices = []
        for (p_s, p_e) in self.padding:
            p_e = None if p_e == 0 else -p_e
            slices.append(slice(p_s, p_e))
        return np_arr[tuple(slices)]
