# Copyright (c) 2019-2020, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4 nowrap:

import typing
import dataclasses as dc
import islpy as isl

import conv
import pipeline as pl
from isl_utils import isl_set_from_names

""" Polyhedral information for operations """


@dc.dataclass(init=False)
class IslAccess:
    """ Wrapper for an isl access mapping: instance space -> object space

    The convention is that the first name of the mapping is the stage, and the
    second is the object (see get_{stage,obj}_name).
    """

    a_ty: str  # RD or WR
    access: isl.Map  # instance space -> object space

    def __init__(self, ty: str, acc: typing.Union[str, isl.Map]):
        if ty not in ("RD", "WR"):
            raise ValueError(
                "Invalid access type: %s. Expecting 'RD' or 'WR'" % (ty,)
            )

        if isinstance(acc, str):
            try:
                acc = isl.Map(acc)
            except:
                print("Failed to create an isl.Map from %s" % (acc,))
                raise

        if not isinstance(acc, isl.Map):
            raise ValueError(
                "Invalid access type: %s. Expecting str or isl.Map"
                % (type(acc),)
            )

        self.a_ty = ty
        self.access = acc

    @staticmethod
    def RD(acc: typing.Union[str, isl.Map]) -> "IslAccess":
        """ Read ISL access """
        return IslAccess("RD", acc)

    @staticmethod
    def WR(acc: typing.Union[str, isl.Map]) -> "IslAccess":
        """ Write ISL access """
        return IslAccess("WR", acc)

    def get_stage_name(self) -> str:
        return self.access.get_tuple_name(isl.dim_type.in_)

    def get_obj_name(self) -> str:
        return self.access.get_tuple_name(isl.dim_type.out)

    def get_idx_ndims(self) -> int:
        return self.access.space.dim(isl.dim_type.in_)

    def get_obj_ndims(self) -> int:
        return self.access.space.dim(isl.dim_type.out)


@dc.dataclass(init=False)
class OpInfo:
    """ Operation (polyhedral) info.

    Object accesses for a given operation

    NB: Each IslAccess is per-object, ie., is a single isl.Map for every object.
    Alternatively, we could use a single isl.UnionMap, but the latter seems
    more complicated.
    """

    op_ty: str
    accesses: typing.List[IslAccess]

    def __init__(self, op_ty: str, accesses: typing.List[IslAccess]):
        self.op_ty = op_ty
        self.accesses = accesses

        # all accesses must have the same stage, and domain
        # (The first check might be redundant because I think having the same
        # domain, implies having the same stage name)
        stage = self.get_stage_name()
        assert all((a.get_stage_name() == stage for a in self.accesses))

        # It would be good if we could verify that all accesses operate in the
        # same domain, but this does not work very well.
        #
        # For example, enabling the code below for the test_conv1d() test fails with:
        # | ValueError: Accesses domains do not match:
        # |   dom:[n, k, p] -> { S1[o1] : k > 0 and 0 <= o1 <= n - k + 2p } acc:IslAccess(a_ty='RD', access=Map("[n, k, p] -> { S1[o1] -> in1[j] : 0 <= o1 <= n - k + 2p and o1 <= j < k + o1 }"))
        # |   dom:[n, k, p] -> { S1[o1] : 0 <= o1 <= n - k + 2p } acc:IslAccess(a_ty='WR', access=Map("[n, k, p] -> { S1[o1] -> in2[j = p + o1] : 0 <= o1 <= n - k + 2p }"))
        #
        # There are probably ways to address this but for now we just check
        # that all the access iterators provide the same index.
        #
        # domain = self.get_domain()
        # if not all((a.access.domain() == domain for a in self.accesses)):
        #     s = "Accesses domains do not match:"
        #     for a in self.accesses:
        #         s += "\n   dom:%s acc:%s" % (a.access.domain(), a)
        #     raise ValueError(s)

    def get_stage_name(self) -> str:
        return self.accesses[0].get_stage_name()

    def get_domain(self) -> isl.Map:
        return self.accesses[0].access.domain()

    def filter_accesses(self, a_ty: str) -> typing.Iterator[IslAccess]:
        return filter(lambda a: a.a_ty == a_ty, self.accesses)

    def rd_accesses(self) -> typing.Iterator[IslAccess]:
        return self.filter_accesses("RD")

    def wr_accesses(self) -> typing.Iterator[IslAccess]:
        return self.filter_accesses("WR")

    def rd_objs(self):
        return (a.get_obj_name() for a in self.rd_accesses())

    def wr_objs(self):
        return (a.get_obj_name() for a in self.wr_accesses())


RD_a = IslAccess.RD
WR_a = IslAccess.WR


def OpInfo_CONV(
    conv_ps: conv.Conv2DParams, s_id: str, vin_id: str, vout_id: str
) -> OpInfo:
    """ OpInfo for a CONV operation """

    rd_a = (
        "{{ {SID}[oh,ow] -> {VID}[id,ih,iw] "
        ":    0   <= oh < {OH} "
        "and  0   <= ow < {OW} "
        "and  0   <= id < {ID} "
        "and  oh  <= ih < oh + {FH} "
        "and  ow  <= iw < ow + {FW} "
        "}}".format(
            ID=conv_ps.i.d,
            OH=conv_ps.o.h,
            OW=conv_ps.o.w,
            FH=conv_ps.f.h,
            FW=conv_ps.f.w,
            SID=s_id,
            VID=vin_id,
        )
    )

    wr_a = (
        "{{ {SID}[oh,ow] -> {VID}[ik,ih,iw] "
        ":    0   <= oh < {OH} "
        "and  0   <= ow < {OW} "
        "and  0   <= ik < {FL} "
        "and  ih = oh + {P} "
        "and  iw = ow + {P} "
        "}}".format(
            OH=conv_ps.o.h,
            OW=conv_ps.o.w,
            FL=conv_ps.f.l,
            P=conv_ps.p_out,
            SID=s_id,
            VID=vout_id,
        )
    )

    return pl.OpInfo("MxV", [RD_a(rd_a), WR_a(wr_a)])


def OpInfo_ADD(
    conv_domain: isl.Map,
    shape: typing.Tuple[int, ...],
    in1_id: str,
    in2_id: str,
    out_id: str,
) -> OpInfo:
    """ OpInfo for an ADD operation """

    (b, d, h, w) = shape
    assert b == 1  # Batch is expected to be 1

    if True:
        # Reconstruct the domain from shape and verify that everyting is in order
        tn = conv_domain.get_tuple_name()
        xdom = isl_set_from_names(tn, ["oh", "ow"])
        xineqs = [  # loop bounds, based on shape
            {1: w - 1, "ow": -1},
            {1: 0, "ow": 1},
            {1: h - 1, "oh": -1},
            {1: 0, "oh": 1},
        ]
        for xineq in xineqs:
            con_ineq = isl.Constraint.ineq_from_names(xdom.space, xineq)
            xdom = xdom.add_constraint(con_ineq)
        # NB: This assertion might eventually fail if we introduce striding or
        # other complications. I just leave it as a sanity check for now.
        assert xdom == conv_domain

    accesses = []
    for (obj_id, mk_acc) in ((in1_id, RD_a), (in2_id, RD_a), (out_id, WR_a)):
        # compute range
        obj_vs = ["%s_%s" % (obj_id, x) for x in ("d", "h", "w")]
        rng = isl_set_from_names(obj_id, obj_vs)
        rel = isl.Map.from_domain_and_range(conv_domain, rng)

        # w,h dimensions
        eqs = [
            {obj_vs[1]: 1, "oh": -1},
            {obj_vs[2]: 1, "ow": -1},
        ]
        for eq in eqs:
            con_eq = isl.Constraint.eq_from_names(rel.space, eq)
            rel = rel.add_constraint(con_eq)
        # d dimension
        ineqs = [
            {1: d - 1, obj_vs[0]: -1},
            {1: 0, obj_vs[0]: 1},
        ]
        for ineq in ineqs:
            con_ineq = isl.Constraint.ineq_from_names(rel.space, ineq)
            rel = rel.add_constraint(con_ineq)
        accesses.append(mk_acc(rel))

    return pl.OpInfo("ADD", accesses)
