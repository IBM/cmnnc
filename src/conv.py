# Copyright (c) 2019-2020, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import typing
import dataclasses as dc

import numpy as np

from util import check_class_hints
from object_info import ObjectInfo

# https://cs231n.github.io/convolutional-networks/


def check_type(v, ty, n=""):
    if not isinstance(v, ty):
        raise TypeError(
            "%s: expecting type %s got type %s" % (n, ty.__name__, type(v))
        )
    return v


""" 2D convolution """


@dc.dataclass(init=False)
class Conv2DInParams:
    w: int
    h: int
    d: int
    """ Convolution input dimensions """

    def __init__(self, *, w, h, d):
        self.w = w
        self.h = h
        self.d = d
        check_class_hints(self)


@dc.dataclass(init=False)
class Conv2DFiltParams:
    """ Convolution filter dimensions """

    w: int
    h: int
    d: int
    l: int

    def __init__(self, *, w, h, d, l):
        self.w = w
        self.h = h
        self.d = d
        self.l = l
        check_class_hints(self)


@dc.dataclass(init=False)
class Conv2DOutParams:
    """ Convolution output dimensions """

    w: int
    h: int
    d: int

    def __init__(self, *, w, h, d):
        self.w = w
        self.h = h
        self.d = d
        check_class_hints(self)

    def to_in(self):
        return Conv2DInParams(w=self.w, h=self.h, d=self.d)


@dc.dataclass(init=False)
class Conv2DParams:
    """ Convolution parameters """

    i: Conv2DInParams
    f: Conv2DFiltParams
    p: int
    p_out: typing.Optional[int]
    s: int
    o: Conv2DOutParams

    def get_out_params(self):
        ow = eval("(i.w - f.w + 2*p) // s + 1", self.__dict__)
        oh = eval("(i.h - f.h + 2*p) // s + 1", self.__dict__)
        od = eval("f.l", self.__dict__)
        return Conv2DOutParams(w=ow, h=oh, d=od)

    def __init__(
        self,
        *,
        i: Conv2DInParams,
        f: Conv2DFiltParams,
        p: int,
        p_out: int,
        s: int
    ):
        self.i = check_type(i, Conv2DInParams, "i")
        self.f = check_type(f, Conv2DFiltParams, "f")
        self.p = p
        self.p_out = p_out
        self.s = s
        self.o = self.get_out_params()
        if self.i.d != self.f.d:
            raise ValueError(
                "input d=%d and filter d=%d parameters do not match",
                (self.i.d, self.f.d),
            )
        check_class_hints(self)

    def get_filters_shape(self):
        """ Get the shape of the filters as a (L,D,H,W) tuple """
        return (self.f.l, self.f.d, self.f.h, self.f.w)

    def get_input_padding(self):
        """ Return something that can be passed to numpy.pad() """
        return ((0, 0), (self.p, self.p), (self.p, self.p))

    def get_output_padding(self):
        """ Return something that can be passed to numpy.pad() """
        return ((0, 0), (self.p_out, self.p_out), (self.p_out, self.p_out))

    def set_p_out_from_padding(self, padding):
        # NB: just handle ((0,0), (p,p), (p,p)) padding for now
        assert padding[0] == (0, 0)
        ((p1, p2), (p3, p4)) = padding[1:]
        (self.p_out) = list(set((p1, p2, p3, p4)))

    def get_input_shape(self, *, pad: bool = False):
        """ Get input shape (pad determines whether padding is considered or not) """
        ph = 2 * self.p if pad else 0
        pw = 2 * self.p if pad else 0
        return (self.i.d, self.i.h + ph, self.i.w + pw)

    def get_input_objectinfo(self):
        return ObjectInfo(self.get_input_shape(), self.get_input_padding())

    def get_output_objectinfo(self):
        return ObjectInfo(self.get_output_shape(), self.get_output_padding())

    def get_output_shape(self, *, pad: bool = False):
        """ Get output shape (pad determines whether padding is considered or not) """
        ph = 2 * self.p_out if pad else 0
        pw = 2 * self.p_out if pad else 0
        return (self.o.d, self.o.h + ph, self.o.w + pw)

    def eval(self, e):
        return eval(e, self.__dict__)


def conv2d_simple(image, filters, conv_params):
    """ Perform a simple CONV (2D) opration

    image: image data (expected shape: conv_ps.get_input_shape())
    filters: filter data (expected shape: conv_ps.get_filters_shape())
    """
    assert image.shape == conv_params.get_input_shape(pad=True), (
        "image.shape=%s different than conv_params.get_input_shape(pad=True):%s"
        % (image.shape, conv_params.get_input_shape(pad=True))
    )
    assert filters.shape == conv_params.get_filters_shape(), (
        "filters.shape=%s different than conv_params.get_filters_shape():%s"
        % (image.shape, conv_params.get_filters_shape())
    )

    output_shape = conv_params.eval("o.d, o.h, o.w")
    output = np.ndarray(output_shape)
    for od in range(conv_params.o.d):
        for oh in range(conv_params.o.h):
            for ow in range(conv_params.o.w):
                iw = conv_params.s * ow
                ih = conv_params.s * oh
                image_block = image[
                    :, ih : ih + conv_params.f.h, iw : iw + conv_params.f.w
                ]
                filter_block = filters[od, :, :, :]
                try:
                    assert image_block.shape == filter_block.shape
                except AssertionError:
                    print(
                        "image_block.shape=",
                        image_block.shape,
                        "filter_block.shape=",
                        filter_block.shape,
                    )
                    raise

                # piecewise multiplication and sum
                output[od][oh][ow] = np.sum(image_block * filter_block)
    return output


def conv2d_mxv(image, filters, conv_params):
    """ Perform a CONV (2D) operation using MxV """

    assert image.shape == conv_params.get_input_shape(pad=True), (
        "image.shape=%s different than conv_params.get_input_shape(pad=True):%s"
        % (image.shape, conv_params.get_input_shape(pad=True))
    )
    assert filters.shape == conv_params.get_filters_shape(), (
        "filters.shape=%s different than conv_params.get_filters_shape():%s"
        % (image.shape, conv_params.get_filters_shape())
    )

    # reshape the filters so that we can use MxV
    filters_m = filters.reshape(conv_params.eval("(f.l, f.d*f.h*f.w)"))
    output_shape = conv_params.get_output_shape()
    output = np.ndarray(output_shape)
    for oh in range(conv_params.o.h):
        for ow in range(conv_params.o.w):
            iw = conv_params.s * ow
            ih = conv_params.s * oh
            image_block = image[
                :, ih : ih + conv_params.f.h, iw : iw + conv_params.f.w
            ]
            image_v = image_block.reshape(conv_params.eval("(f.d*f.h*f.w)"))
            res = np.matmul(filters_m, image_v)
            # print("m=", filters_m.shape, "v=", image_v.shape)

            output[:, oh, ow] = res
    return output


""" 1D Convolution """


@dc.dataclass(init=False)
class Conv1DInParams:
    w: int
    d: int
    """ Convolution input dimensions """

    def __init__(self, *, w, d):
        self.w = w
        self.d = d


@dc.dataclass(init=False)
class Conv1DFiltParams:
    """ Convolution filter dimensions """

    w: int
    d: int
    l: int

    def __init__(self, *, w, d, l):
        self.w = w
        self.d = d
        self.l = l


@dc.dataclass(init=False)
class Conv1DOutParams:
    """ Convolution output dimensions """

    w: int
    d: int

    def __init__(self, *, w, d):
        self.w = w
        self.d = d

    def to_in(self):
        return Conv1DInParams(w=self.w, d=self.d)


@dc.dataclass(init=False)
class Conv1DParams:
    """ Convolution parameters """

    i: Conv1DInParams
    f: Conv1DFiltParams
    p: int
    s: int
    o: Conv1DOutParams

    def eval(self, e):
        return eval(e, self.__dict__)

    def __init__(
        self,
        *,
        i: Conv1DInParams,
        f: Conv1DFiltParams,
        p: int,
        p_out: int,
        s: int
    ):
        self.i = check_type(i, Conv1DInParams, "i")
        self.f = check_type(f, Conv1DFiltParams, "i")
        self.p = p
        self.p_out = p_out
        self.s = s
        self.o = self.get_out_params()
        if self.i.d != self.f.d:
            raise ValueError(
                "input d=%d and filter d=%d parameters do not match",
                (self.i.d, self.f.d),
            )

    def get_out_params(self):
        ow = eval("(i.w - f.w + 2*p) // s + 1", self.__dict__)
        od = eval("f.l", self.__dict__)
        return Conv1DOutParams(w=ow, d=od)

    def get_filters_shape(self):
        """ Get the shape of the filters as a (L,D,W) tuple """
        return (self.f.l, self.f.d, self.f.w)

    def get_input_padding(self):
        """ Return something that can be passed to numpy.pad() """
        return ((0, 0), (self.p, self.p))

    def get_output_padding(self):
        """ Return something that can be passed to numpy.pad() """
        return ((0, 0), (self.p_out, self.p_out))

    def get_input_shape(self, *, pad: bool = False):
        """ Get input shape (pad determines whether padding is considered or not) """
        pw = 2 * self.p if pad else 0
        return (self.i.d, self.i.w)

    def get_output_shape(self, *, pad: bool = False):
        """ Get output shape (pad determines whether padding is considered or not) """
        pw = 2 * self.p_out if pad else 0
        return (self.o.d, self.o.w + pw)

    def get_input_objectinfo(self):
        return ObjectInfo(self.get_input_shape(), self.get_input_padding())

    def get_output_objectinfo(self):
        return ObjectInfo(self.get_output_shape(), self.get_output_padding())


def conv1d_simple(image, filters, params: Conv1DParams):
    output_shape = params.eval("o.d, o.w")
    output = np.ndarray(output_shape)
    for od in range(params.o.d):
        for ow in range(params.o.w):
            iw = params.s * ow
            image_block = image[:, iw : iw + params.f.w]
            filter_block = filters[od, :, :]
            try:
                assert image_block.shape == filter_block.shape
            except AssertionError:
                print(
                    "ow=",
                    ow,
                    "image.shape=",
                    image.shape,
                    "image_block.shape=",
                    image_block.shape,
                    "filter_block.shape=",
                    filter_block.shape,
                )
                raise

            # piecewise multiplication and sum
            output[od][ow] = np.sum(image_block * filter_block)
    return output
