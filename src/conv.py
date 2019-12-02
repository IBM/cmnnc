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

    def to_in(self):
        return Conv2DInParams(w=self.w, h=self.h, d=self.d)

@dc.dataclass(init=False)
class Conv2DParams:
    """ Convolution parameters """
    i: Conv2DInParams
    f: Conv2DFiltParams
    p: int
    s: int
    o: Conv2DOutParams

    def get_out_params(self):
        ow = eval("(i.w - f.w + 2*p) // s + 1", self.__dict__)
        oh = eval("(i.h - f.h + 2*p) // s + 1", self.__dict__)
        od = eval("f.l", self.__dict__)
        return Conv2DOutParams(w=ow, h=oh, d=od)

    def __init__(self, *,  i: Conv2DInParams, f: Conv2DFiltParams, p: int, p_out: int, s: int):
        self.i = check_type(i, Conv2DInParams, "i")
        self.f = check_type(f, Conv2DFiltParams, "f")
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
        # XXX: Don't we need to consider p_out here?
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

    def get_rd_a(self, *, s_id, vin_id):
        rd_a = \
            "{{ S{SID}[oh,ow] -> V{VID}[id,ih,iw] " \
            ":    0   <= oh < {OH} " \
            "and  0   <= ow < {OW} " \
            "and  0   <= id < {ID} " \
            "and  oh  <= ih < oh + {FH} "\
            "and  ow  <= iw < ow + {FW} "\
            "}}" \
            .format(
                ID=self.i.d,
                OH=self.o.h, OW=self.o.w,
                FH=self.f.h, FW=self.f.w,
                SID=s_id, VID=vin_id
            )
        return rd_a

    def get_wr_a(self, *, s_id, vout_id):
        wr_a = \
            "{{ S{SID}[oh,ow] -> V{VID}[ik,ih,iw] " \
            ":    0   <= oh < {OH} " \
            "and  0   <= ow < {OW} " \
            "and  0   <= ik < {FL} " \
            "and  ih = oh + {P} " \
            "and  iw = ow + {P} " \
            "}}" \
            .format(
                OH=self.o.h, OW=self.o.w,
                FL=self.f.l,
                P=self.p_out,
                SID=s_id, VID=vout_id
            )
        return wr_a

    def get_rd_wr_a(self, *, s_id, vin_id, vout_id):
        """ Return read and write access relations """
        rd_a = self.get_rd_a(s_id=s_id, vin_id=vin_id)
        wr_a = self.get_wr_a(s_id=s_id, vout_id=vout_id)
        return (rd_a, wr_a)

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

    def __init__(self, *, i: Conv1DInParams, f: Conv1DFiltParams, p: int, p_out: int, s: int):
        self.i = check_type(i, Conv1DInParams, "i")
        self.f = check_type(f, Conv1DFiltParams, "i")
        self.p = p
        self.p_out = p_out
        self.s = s
        self.o = self.get_out_params()
        if self.i.d != self.f.d:
            raise ValueError("input d=%d and filter d=%d parameters do not match", (self.i.d, self.f.d))

    def get_out_params(self):
        ow = eval("(i.w - f.w + 2*p) // s + 1", self.__dict__)
        od = eval("f.l", self.__dict__)
        return Conv1DOutParams(w=ow, d=od)

    def get_filters_shape(self):
        """ Get the shape of the filters as a (L,D,W) tuple """
        return (self.f.l, self.f.d, self.f.w)

    def get_image_shape(self):
        """ Get the shape of the image (no padding) as a (D,W) tuple """
        return (self.i.d, self.i.w)

    def get_padding(self):
        """ Return something that you can pass to numpy.pad() """
        return ((0,0), (self.p, self.p))

    def get_in_shape(self):
        """ Get the input shape (including padding) as a (D,W) tuple """
        return (self.i.d, self.i.w + 2*self.p)

    def get_out_shape(self):
        """ Get the output shape as a (D,H,W) tuple """
        # XXX: Don't we need to consider p_out, here?
        return (self.o.d, self.o.w)

def conv1d_simple(image, filters, params: Conv1DParams):
    output_shape = params.eval("o.d, o.w")
    output = np.ndarray(output_shape)
    for od in range(params.o.d):
        for ow in range(params.o.w):
            iw = params.s*ow
            image_block = image[:,iw: iw + params.f.w]
            filter_block = filters[od,:,:]
            try:
                assert image_block.shape == filter_block.shape
            except AssertionError:
                print("image_block.shape=", image_block.shape, "filter_block.shape=", filter_block.shape)
                raise

            # piecewise multiplication and sum
            output[od][ow] = np.sum(image_block*filter_block)
    return output

