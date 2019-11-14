# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import dataclasses as dc

import numpy as np

# https://cs231n.github.io/convolutional-networks/

def check_type(v, ty, n=""):
    if not isinstance(v, ty):
        raise TypeError("%s: expecting type %s got type %s" % (n, ty.__name__, type(v)))
    return v

@dc.dataclass(init=False)
class ConvInParams:
    """ Convolution input dimensions """
    def __init__(self, *, w, h, d):
        self.w = w
        self.h = h
        self.d = d

@dc.dataclass(init=False)
class ConvFiltParams:
    """ Convolution filter dimensions """
    def __init__(self, *, w, h, d, l):
        self.w = w
        self.h = h
        self.d = d
        self.l = l

@dc.dataclass(init=False)
class ConvOutParams:
    """ Convolution output dimensions """
    def __init__(self, *, w, h, d):
        self.w = w
        self.h = h
        self.d = d

    def to_in(self):
        return ConvInParams(w=self.w, h=self.h, d=self.d)

class ConvParams:
    """ Convolution parameters """
    i: ConvInParams
    f: ConvFiltParams
    p: int
    s: int
    o: ConvOutParams

    def get_out_params(self):
        ow = eval("(i.w - f.w + 2*p) // s + 1", self.__dict__)
        oh = eval("(i.h - f.h + 2*p) // s + 1", self.__dict__)
        od = eval("f.l", self.__dict__)
        return ConvOutParams(w=ow, h=oh, d=od)

    def __init__(self, *,  i: ConvInParams, f: ConvFiltParams, p: int, p_out: int, s: int):
        self.i = check_type(i, ConvInParams, "i")
        self.f = check_type(f, ConvFiltParams, "f")
        self.p = p
        self.p_out = p_out
        self.s = s
        self.o = self.get_out_params()
        if self.i.d != self.f.d:
            raise ValueError("input d=%d and filter d=%d parameters do not match", (self.i.d, self.f.d))

    def get_in_shape(self):
        """ Get the input shape (including padding) as a (D,H,W) tuple """
        return (self.i.d, self.i.h + 2*self.p, self.i.w + 2*self.p)

    def get_out_shape(self):
        """ Get the output shape as a (D,H,W) tuple """
        return (self.o.d, self.o.h, self.o.w)

    def get_filters_shape(self):
        """ Get the shape of the filters as a (L,D,H,W) tuple """
        return (self.f.l, self.f.d, self.f.h, self.f.w)

    def get_image_shape(self):
        """ Get the shape of the image (no padding) as a (D,H,W) tuple """
        return (self.i.d, self.i.h, self.i.w)

    def get_padding(self):
        """ Return something that you can pass to numpy.pad() """
        return ((0,0), (self.p, self.p), (self.p, self.p))

    def eval(self, e):
        return eval(e, self.__dict__)


def conv2d_simple(image, filters, conv_params):
    output_shape = conv_params.eval("o.d, o.h, o.w")
    output = np.ndarray(output_shape)
    for od in range(conv_params.o.d):
        for oh in range(conv_params.o.h):
            for ow in range(conv_params.o.w):
                iw = conv_params.s*ow
                ih = conv_params.s*oh
                image_block  = image[:,ih:ih + conv_params.f.h, iw:iw + conv_params.f.w]
                filter_block = filters[od,:,:,:]
                try:
                    assert image_block.shape == filter_block.shape
                except AssertionError:
                    print("image_block.shape=", image_block.shape, "filter_block.shape=", filter_block.shape)
                    raise

                # piecewise multiplication and sum
                output[od][oh][ow] = np.sum(image_block*filter_block)
    return output

def conv2d_mxv(image, filters, conv_params):
    # compute the matrix based on the filters

    # reshape the filters so that we can use MxV
    filters_m = filters.reshape(conv_params.eval("(f.l, f.d*f.h*f.w)"))
    output_shape = conv_params.get_out_shape()
    output = np.ndarray(output_shape)
    for oh in range(conv_params.o.h):
        for ow in range(conv_params.o.w):
            iw = conv_params.s*ow
            ih = conv_params.s*oh
            image_block  = image[:,ih:ih + conv_params.f.h, iw:iw + conv_params.f.w]
            image_v = image_block.reshape(conv_params.eval("(f.d*f.h*f.w)"))
            res = np.matmul(filters_m, image_v)
            # print("m=", filters_m.shape, "v=", image_v.shape)

            output[:,oh,ow] = res
    return output

import unittest

class TestConv(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # same parameters as CIFAR's first convolution
        conv_ps = self.conv_ps = ConvParams(
            i = ConvInParams(w=32, h=32, d=3),
            f = ConvFiltParams(w=3, h=3, d=3, l=16),
            p = 1,
            s = 1,
            p_out = 0
        )

        # Initalize random filtrs and image
        self.filters = np.random.rand(*conv_ps.get_filters_shape())
        self.image   = np.random.rand(*conv_ps.get_image_shape())
        self.image   = np.pad(self.image, conv_ps.get_padding())

    def test_mxv(self):
        """ Test MxV and simple versions of CONV 2D """
        output_simple = conv2d_simple(self.image, self.filters, self.conv_ps)
        output_mxv = conv2d_mxv(self.image, self.filters, self.conv_ps)
        np.testing.assert_allclose(output_simple, output_mxv)
