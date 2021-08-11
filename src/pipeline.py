# Copyright (c) 2019-2020, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

# pylint: disable-msg=invalid-name
# pylint: disable-msg=missing-docstring
# pylint: disable-msg=no-else-return
# pylint: disable-msg=too-few-public-methods
# pylint: disable-msg=too-many-locals
# pylint: disable-msg=too-many-branches
# pylint: disable-msg=too-many-instance-attributes
# pylint: disable-msg=protected-access
# pylint: disable-msg=line-too-long
# pylint: disable-msg=no-else-break
# pylint: disable-msg=attribute-defined-outside-init


""" Pipeline execution

The pipeline consists of stages that are executed on cores.

"""

import itertools
import typing
import types
import dataclasses as dc
import ast as pyast
import pprint as pp
from collections import deque

import numpy as np
import astor as pyastor

import islpy as isl

from op_info import OpInfo
from object_info import ObjectInfo
from pyast_utils import StructureTupleYields
from util import check_class_hints
from isl_utils import (
    isl2py_fn,
    isl_map_to_ast,
    isl_rel_loc_to_max_iter,
    isl_set_from_names,
    isl_set_from_shape,
    isl_fix_params,
)


@dc.dataclass(init=False)
class StageInfo:
    """ Polyhedral information for a pipeline stage

    The idea able to express the code for every stage in a fused loop as
    follows:
      for i in ...
          for j in ...
              ....
                  MxV()
                  DPU_INS1()
                  DPU_INS2()
                  ....

    Each operation (MXV, DPU_INS) has a bunch of read/write accesses to objects
    that are represented as polyhedral access relations.

    As we add operations, we maintain unconnected inputs and outputs. (An
    unconnected input is an object that is read, but not written within this
    stage, and vice-versa for unconnected outputs). These are the read and
    write access relations for a stage.
    """

    ops: typing.List[OpInfo]
    ro_objs: typing.Set[str]  # Objects that this stage reads (external)
    wo_objs: typing.Set[str]  # Objects that this stage writes (external)
    rw_objs: typing.Set[str]  # Objects that this stage writes and reads (internal)

    def __init__(self, ops: typing.List[OpInfo]):
        if len(ops) == 0:
            raise ValueError("No operations provided (length is 0)")
        if ops[0].op_ty != "MxV":
            # NB: This used to throw a RuntimeError, but there is at least one
            # test currently that uses a sinlge ID operation (no MxV)
            print("First operation is not MxV (instead is:%s)" % (ops[0].op_ty))
        self.ops = ops

        # all operations must have the same stage, and the same domain
        # (The first check might be redundant because I think having the same
        # domain, implies having the same stage name)
        stage = self.get_stage_name()
        assert all((o.get_stage_name() == stage for o in ops))
        domain = self.get_domain()
        assert all((o.get_domain() == domain for o in ops))

        self.ro_objs = set()
        self.wo_objs = set()
        self.rw_objs = set()

        # process object dependencies, do sanity checks, and create object sets
        for (op_id, op) in enumerate(self.ops):
            for acc in op.accesses:
                objname = acc.get_obj_name()
                if acc.a_ty == "RD":
                    if objname in self.wo_objs:
                        # Object was previously written, no read. Move it to rw
                        self.wo_objs.remove(objname)
                        self.rw_objs.add(objname)
                    elif objname in self.ro_objs:
                        # Object was previously read, do nothing
                        pass
                    elif objname in self.rw_objs:
                        # Object was previously written and read, do nothing
                        pass
                    else:
                        # First time seeing this object, put it into ro set
                        self.ro_objs.add(objname)
                elif acc.a_ty == "WR":
                    if objname in self.wo_objs:
                        # Object was previously written, error
                        raise ValueError(
                            "Object %s written in op %s (id:%d) but also written previously"
                            % (objname, op.op_ty, op_id)
                        )
                    if objname in self.ro_objs:
                        raise ValueError(
                            "Object %s written in op %s (id:%d) previously read"
                            % (objname, op.op_ty, op_id)
                        )
                    if objname in self.ro_objs:
                        raise ValueError(
                            "Object %s written in op %s (id:%d) previously write and read"
                            % (objname, op.op_ty, op_id)
                        )
                    self.wo_objs.add(objname)
                else:
                    ValueError("Unknown access type: %s" % (acc.a_ty,))

    def get_stage_name(self) -> str:
        return self.ops[0].get_stage_name()

    def get_domain(self) -> isl.Map:
        return self.ops[0].get_domain()

    def get_obj_rd_rel(self, objname: str) -> isl.Map:
        if objname not in self.ro_objs:
            raise ValueError(
                "stage %s does not read from object %s"
                % (self.get_stage_name(), objname)
            )
        accs = []
        for op in self.ops:
            accs.extend(
                a for a in op.rd_accesses() if a.get_obj_name() == objname
            )
        acc0 = accs[0]
        if len(accs) > 1:
            # Combine multiple read accesses for an object
            for acc in accs[1:]:
                acc0.access = acc0.access.union(acc.access)

        return acc0.access

    def get_obj_wr_rel(self, objname: str) -> isl.Map:
        if objname not in self.wo_objs:
            raise ValueError(
                "stage %s does not write to object %s"
                % (self.get_stage_name(), objname)
            )
        accs = []
        for op in self.ops:
            accs.extend(
                a for a in op.wr_accesses() if a.get_obj_name() == objname
            )
        assert len(accs) == 1, "Multiple writes on the same object not allowed"
        return accs[0].access

    def get_obj_last_loc(self, objname: str, param_vals) -> isl.Map:
        if objname not in self.wo_objs:
            raise ValueError(
                "stage %s does not write to object %s"
                % (self.get_stage_name(), objname)
            )
        accs = []
        for op in self.ops:
            accs.extend(
                a for a in op.wr_accesses() if a.get_obj_name() == objname
            )
        assert len(accs) == 1, "Multiple writes on the same object not allowed"
        acc = accs[0].access
        last_loc = isl_fix_params(acc.domain().lexmax(), param_vals)
        ps = []
        last_loc.foreach_point(ps.append)
        (p,) = ps
        loc = tuple(
            p.get_coordinate_val(isl.dim_type.all, i).to_python()
            for i in range(len(p.get_id_dict()))
        )
        return loc


def isl_map_to_pyfn(rel, fnname, s=None):
    """ Transform an isl map to a python function """
    ast = isl_map_to_ast(rel)
    py = isl2py_fn(ast, fnname)

    # isl_map_to_ast works by generating code for rel.unwrap() The resulting
    # python function returns a tuple containing items from the domain and the
    # image of the relationship, but there is no way to distinguish which is
    # which.
    #
    # Here, we apply a transformation that structures the yields of the
    # generated function so that they yield a 2-tuple of tuples, one for the
    # doimain and one for the image.
    if s is None:
        s = (_nin, _nout) = tuple(
            rel.space.dim(x) for x in (isl.dim_type.in_, isl.dim_type.out)
        )
    StructureTupleYields(s).visit(py)
    return py


def rel_a_iter(rel_iter):
    # Accesses iterators (si.rd_a, and si.wr_a) iterate the relation from
    # indices to object locations. However, an iteration might require
    # multiple locations (usually for reads).
    #
    # This is an adapter that groups all accesses for a single iteration
    # into a list, and yields the iteration index with the access list
    last_idx = None
    l = []
    for ri in rel_iter():
        try:
            (idx, loc) = ri
        except ValueError:
            print(
                "%s: ri=%s cannot be packed into (idx,loc)"
                % (rel_iter.__name__, ri,)
            )
            raise
        if idx == last_idx:
            l.append(loc)
        else:
            if len(l) != 0:
                yield (last_idx, l)
            l = [loc]
            last_idx = idx
    if len(l) > 0:
        yield (last_idx, l)


@dc.dataclass(init=False)
class ExecOp:
    ty: str
    accesses: typing.Dict[
        str,
        typing.Dict[str, typing.Optional[typing.List[typing.Tuple[int, ...]]]],
    ]

    def __init__(self, ty, rd_objs, wr_objs):
        self.ty = ty
        self.accesses = {}
        self.accesses["RD"] = dict((o, None) for o in rd_objs)
        self.accesses["WR"] = dict((o, None) for o in wr_objs)


class AccessIterator:
    idx: typing.Tuple[int, ...]
    ops: typing.List[ExecOp]

    def __init__(self, stage):
        self.idx_ = None
        self.ops = [
            ExecOp(ty=op.op_ty, rd_objs=op.rd_objs(), wr_objs=op.wr_objs())
            for op in stage.si.ops
        ]
        self.stage_name = stage.get_name()  # For debugging messages
        self.fns = []

    def set_access(self, op_id, a_ty, a_obj, a_list):
        op = self.ops[op_id]
        assert op.accesses[a_ty][a_obj] is None
        op.accesses[a_ty][a_obj] = a_list

    def reset_access(self):
        for op in self.ops:
            for ty in ("RD", "WR"):
                for obj in op.accesses[ty]:
                    op.accesses[ty][obj] = None
        self.idx_ = None

    def register_iter_fn(self, op_id, op_ty, a_ty, obj):
        def update_state_dec(fn):
            for (idx, access_l) in fn:
                # print("%s: LOOP: => idx_=%s idx=%s" %(self.stage_name, self.idx_, idx))
                # Verify that all iterators operate on the same domain by
                # checking that they produced the same index
                if self.idx_ is None:
                    self.idx_ = idx
                else:
                    assert (
                        self.idx_ == idx
                    ), "Expecting idx=%s but got idx=%s" % (self.idx_, idx,)
                self.set_access(op_id, a_ty, obj, access_l)
                yield

        def decorator(fn):
            wrapped_fn = lambda: update_state_dec(rel_a_iter(fn))
            self.fns.append(wrapped_fn)
            return fn

        return decorator

    def loop(
        self, inp_limit: typing.Optional[int] = None
    ) -> typing.Iterator[typing.Tuple[typing.Tuple[int, ...], typing.List[ExecOp]]]:
        for inp in itertools.count():
            if inp_limit is not None and inp >= inp_limit:
                break
            iters = [f() for f in self.fns]
            niters = 0
            for _ in zip(*iters):
                # print("%s: LOOP => %s " % (self.stage_name, self.idx_))
                niters += 1
                yield ((inp,) + self.idx_, self.ops)
                self.reset_access()
            assert niters > 0


class LocToMaxIterIterator:
    def __init__(self, stage):
        # Maximum iteration allowed for every object that needs to be read
        #  if maximum iteration is None, no writes have happened yet.
        self.obj_max_iter = dict((o, None) for o in stage.si.ro_objs)

        # Initialized generators
        self.max_iter_gs = {}

        self.stage_name = stage.get_name()  # debugging

    def register_iter_fn(self, obj):
        def max_iter_gen(fn):
            """ Python generator that consumers writes and updates self.obj_max_iter """

            print(
                "%s: Initializing max_iter_gen generator for object: %s (%s)"
                % (self.stage_name, obj, fn)
            )
            # inp  is the id of the input (typically image) being processed
            # This is used to maintain proper ordering when we wrap-around
            for inp in itertools.count():
                for (write, max_iter) in fn():
                    while True:
                        new_write = yield
                        if new_write == write:
                            self.obj_max_iter[obj] = (inp,) + max_iter
                            print(
                                "%s:\tGot expected write: %s. max_iter is now: %s"
                                % (
                                    self.stage_name,
                                    new_write,
                                    self.obj_max_iter[obj],
                                )
                            )
                            break
                        else:
                            print(
                                "%s:\tGot %s, but expecting %s to change max_iter"
                                % (self.stage_name, new_write, write)
                            )

        def decorator(fn):
            # Initialize generator
            gen = max_iter_gen(fn)
            gen.send(None)
            assert obj not in self.max_iter_gs
            self.max_iter_gs[obj] = gen
            return fn

        return decorator

    def set_dont_wait_for_reads(self, objname):
        del self.obj_max_iter[objname]

    def handle_write(self, obj, wr_idx):
        # Call the generator that consumes writes
        self.max_iter_gs[obj].send(wr_idx)

    def reads_ready(self, idx):
        """ Return whether the reads for iteration i are ready """

        def obj_rdy(idx, obj: str, max_iter) -> bool:
            if max_iter is None:
                print(
                    "%s: max_iter for object %s unset. Reads not ready"
                    % (self.stage_name, obj)
                )
                return False
            assert isinstance(max_iter, tuple)
            assert len(idx) == len(max_iter)
            return idx <= max_iter

        # NB: if the iteratable is empty, all() returns true
        return all(
            obj_rdy(idx, o, max_i) for (o, max_i) in self.obj_max_iter.items()
        )


class Stage:
    def __init__(self, si: StageInfo, param_vals=None):
        """ Initialize a stage

        si: StageInfo for this stage
        param_vals: dict of values for parameters in SI experessions (or None).

        The @param_vals argument is passed to the execution of the generated
        python modules.
        """
        self.si = si
        self.param_vals = param_vals if param_vals is not None else dict()
        self.print_ast_ = False

        # For every object this stage needs to read, this dict stores the ISL
        # relation that provides the following mapping:
        #  observed object writes to the maximum iteration that can be executed.
        # This relation is set by the pipeline
        self.loctomaxiter_rel = dict((o, None) for o in self.si.ro_objs)

        # Helpers for combining generated iterators
        #  They provide decorators that are added in the generated code
        self.access_i = AccessIterator(self)
        self.loctomaxiter_i = LocToMaxIterIterator(self)

    def get_ro_objnames(self) -> typing.Set[str]:
        """ Return the objects that this stage reads (only) """
        return self.si.ro_objs

    def get_wo_objnames(self) -> typing.Set[str]:
        """ Return the objects that this stage writes (only) """
        return self.si.wo_objs

    def get_rw_objnames(self) -> typing.Set[str]:
        """ Return the objects that this stage both writes and reads.

        These are objects internal in the stage.
        """
        return self.si.rw_objs

    def attach_to_pipeline(self, pipeline_write, execute_ops):
        self.pipeline_write = pipeline_write
        self.core = Core()
        self.execute_ops = execute_ops

    def set_isl_rel_loc_to_max_iter(self, objname: str, rel: isl.Map):
        """ set the relation to compute the maximum iteration based on writes """
        if self.loctomaxiter_rel[objname] is not None:
            raise ValueError(
                "loctomaxiter_rel alredy set for object %s" % (objname,)
            )
        self.loctomaxiter_rel[objname] = rel

    def set_dont_wait_for_reads(self, objname):
        """ Remove objname from obj_max_iter so that we never wait for reads on this object """
        if self.loctomaxiter_rel[objname] is not None:
            raise ValueError(
                "loctomaxiter_rel alredy set for object %s" % (objname,)
            )
        self.loctomaxiter_i.set_dont_wait_for_reads(objname)

    def build_module(self):
        self.pymod = self.build_module_()

    def build_module_(self):
        body = []

        for (op_id, op) in enumerate(self.si.ops):
            # Generate individual functions for every access
            #
            # NB: I tried using islMap.range_product() for creating a single
            # iterator for all accesses, but it did not work because there
            # might be more than one data accesses per iteration for a given
            # object. For more details, check src/test_range_product_gen.py
            #
            # If we could have the IST AST generate a single data access
            # description for all iterations, then we can revisit this.
            for acc in op.accesses:
                fn_name = "%s_%02d_%s_%s_%s" % (
                    self.get_name(),
                    op_id,
                    op.op_ty,
                    acc.a_ty.lower(),
                    acc.get_obj_name(),
                )
                py = isl_map_to_pyfn(acc.access, fn_name)
                s = "access_iter(op_id=%d, op_ty='%s', a_ty='%s', obj='%s')" % (
                    op_id,
                    op.op_ty,
                    acc.a_ty,
                    acc.get_obj_name(),
                )
                dec_call = pyast.parse(s).body[0].value
                py.decorator_list.append(dec_call)
                body.append(py)

        # generate loc_to_max_iter() functions for every object read by this stage
        for (obj, rel) in self.loctomaxiter_rel.items():
            if rel is not None:
                fnname = "%s_%s_loc_to_max_iter" % (self.get_name(), obj)
                py = isl_map_to_pyfn(rel, fnname)
                s = "loc_to_maxiter_iter(obj='%s')" % (obj,)
                dec_call = pyast.parse(s).body[0].value
                py.decorator_list.append(dec_call)
                body.append(py)

        ast_mod = pyast.Module(body=body)
        if self.print_ast_:
            s = "Module for %s" % (self.get_name(),)
            print("-" * 10, s, "-" * (80 - 10 - len(s) - 2))
            # print(astpp_dump(ast_mod))
            # print("-"*80)
            print(pyastor.to_source(ast_mod))
            print("-" * 80)

        pyast.fix_missing_locations(ast_mod)
        code = compile(ast_mod, "<generated>", "exec")
        ret = types.ModuleType("stage_%s" % (self.get_name()))
        ret.__dict__.update(self.param_vals)
        ret.__dict__.update({"access_iter": self.access_i.register_iter_fn})
        ret.__dict__.update(
            {"loc_to_maxiter_iter": self.loctomaxiter_i.register_iter_fn}
        )
        exec(code, ret.__dict__)
        return ret

    def get_name(self):
        return self.si.get_stage_name()

    def get_wr_obj_name(self):
        return self.si.get_wr_obj_name()

    def reads_ready(self, idx):
        """ Return whether the reads for iteration i are ready """
        return self.loctomaxiter_i.reads_ready(idx)

    def issue_write(self, wr_obj, wr_idx, wr_val):
        # print("%s: Issuing write: obj:%s idx:%s" % (self.get_name(), wr_obj, wr_idx, wr_val))
        if self.pipeline_write is not None:
            self.pipeline_write(self, wr_obj, wr_idx, wr_val)

    def write_callback(self, wr_objstr, wr_idx, wr_val):
        """ Write callback executed on the reader

        Writes the data to the core-local object and executes the "snooping for
        SRAM writes" logic.
        """
        print("%s: Callback on write: wr_obj:%s wr_idx:%s wr_val:%s"
             % (self.get_name(), wr_objstr, wr_idx, wr_val))

        # the write should be in the object that we read
        assert wr_objstr in self.si.ro_objs

        # Write data (if a value exists)
        if wr_val is not None:
            self.core.write_obj(wr_objstr, wr_idx, wr_val)
        else:
            self.core.validate_write(wr_objstr, wr_idx)

        self.loctomaxiter_i.handle_write(wr_objstr, wr_idx)

    def tick_gen(self, loop_inp_limit=None):
        """ Tick generator: executes a single tick

        yields the iteration executed or None if it was stalled
        """
        for (idx, ops) in self.access_i.loop(loop_inp_limit):
            while not self.reads_ready(idx):
                print("%s: Stalling iteration %s." % (self.get_name(), idx))
                yield None

            print("%s: Executing iteration %s." % (self.get_name(), idx))
            if self.execute_ops:
                out = self.core.execute_ops(ops)
            else:
                out = self.core.validate_ops(ops)

            for (objstr, objvals) in out.items():
                for (wr_i, wr_v) in objvals:
                    self.issue_write(objstr, wr_i, wr_v)

            yield idx

    def __repr__(self):
        return "Stage(%s)" % (self.si,)


class CoreConf:
    """ Core configuration """

    def __init__(self, xbar_m: np.ndarray):
        """ Intialize core configuration """
        self.xbar_m = xbar_m


class Core:
    """ Core: crossbar and digital unit """

    width: int = 256  #
    xbar_m: typing.Optional[np.ndarray]
    # NB: For now, we just keep objects as np arrays. Eventually, we might want
    # to map them to a linear buffer representing the core's SRAM.
    objs: typing.Dict[str, np.ndarray]
    objs_info: typing.Dict[str, ObjectInfo]

    def __init__(self):
        self.xbar_m = None
        self.objs = {}
        self.objs_info = {}
        self.internal_objs = {}

    def configure(self, cnf: CoreConf):
        self.set_xbar_matrix(cnf.xbar_m)

    def set_xbar_matrix(self, xbar_m):
        (xbar_m_width, xbar_m_height) = xbar_m.shape
        # we accept whatever matrix we are given, as long as it fits into the
        # crossbar.
        if xbar_m_width > self.width or xbar_m_height > self.width:
            raise ValueError(
                "XBAR too small: XBAR width is %d, while given matrix shape is:%s"
                % (self.width, xbar_m.shape)
            )
        self.xbar_m = xbar_m.copy()

    def alloc_object(self, objname: str, info: ObjectInfo):
        padded_shape = info.get_padded_shape()
        print("Allocating %s (padded_shape:%s)" % (objname, padded_shape))
        obj = np.zeros(padded_shape)
        self.set_object(objname, obj, info)

    def set_object(self, objname: str, obj: np.ndarray, info: ObjectInfo):
        if objname in self.objs:
            raise ValueError("object %s already exists" % (objname,))
        self.objs[objname] = obj
        if objname in self.objs_info:
            raise ValueError("info for object %s already exists" % (objname,))
        self.objs_info[objname] = info

    def get_object(self, objname: str):
        return self.objs[objname]

    def alloc_internal_object(self, objname: str, info: ObjectInfo):
        padded_shape = info.get_padded_shape()
        obj = np.zeros(padded_shape)
        self.set_internal_object(objname, obj, info)

    def set_internal_object(
        self, objname: str, obj: np.ndarray, info: ObjectInfo
    ):
        if objname in self.internal_objs:
            raise ValueError("internal object %s already exists" % (objname,))
        self.internal_objs[objname] = obj
        if objname in self.objs_info:
            raise ValueError("info for object %s already exists" % (objname,))
        self.objs_info[objname] = info

    def get_internal_object(self, objname: str):
        return self.internal_objs[objname]

    def validate_ops(self, ops: typing.List[ExecOp]):
        """ Valdidate operations

        Ensure that the reads are within the array bounds.
        Will raise an error if that's not the case
        """

        # TODO: Use self.read_object()
        ret = {}
        for op in ops:
            for (rd_objstr, rd_is) in op.accesses["RD"].items():
                if rd_objstr not in self.objs:
                    raise ValueError(
                        "object %s does not exist in this core" % (rd_objstr,)
                    )
                obj = self.objs[rd_objstr]
                for idx in rd_is:
                    assert isinstance(idx, tuple) and len(idx) == len(
                        obj.shape
                    ), "idx=%s obj.shape=%s" % (idx, obj.shape)
                    try:
                        _ = obj[idx]
                    except:
                        print(
                            "Failed to access %s (shape=%s) on %s"
                            % (rd_objstr, obj.shape, idx)
                        )
                        raise
                for (wr_objstr, wr_is) in op.accesses["WR"].items():
                    assert wr_objstr not in ret
                    ret[wr_objstr] = zip(wr_is, itertools.repeat(None))

        return ret

    def read_object(self, objstr, rd_is, unpad_oi=None):
        """ Read data from object local to this core

        objstr: name of th eobject
        rd_is: indices to read
        unpad_oi: object info if we want to "unpad" the object, or None
        """
        # An object is either a core-local object or an intermediate result from this set of operations.
        if objstr in self.objs:
            obj = self.objs[objstr]
        elif objstr in self.internal_objs:
            obj = self.internal_objs[objstr]
        else:
            raise ValueError(
                "object %s not found in local objects (%s) or intermediate results (%s)"
                % (objstr, ",".join(self.objs), ",".join(self.internal_objs))
            )

        if unpad_oi is not None:
            obj = unpad_oi.get_unpadded_slice(obj)

        # NB: not sure if we need to deal with multi-dimensional objects or how
        # to. For now assume that objects stored on cores are 1D
        ret = np.zeros(shape=(len(rd_is),))
        for i, idx in enumerate(rd_is):
            # We check if there is a shape attribute to accomodate the stupid way we deal with intermediate results.
            # Once we fix this, we can remove the check.
            assert isinstance(idx, tuple) and (
                getattr(obj, "shape", None) is None
                or len(idx) == len(obj.shape)
            ), "idx=%s obj.shape=%s" % (idx, obj.shape)
            ret[i] = obj[idx]
        return ret

    def handle_op_output(
        self,
        objstr: str,
        results: typing.Dict[str, np.ndarray],
        wr_is: typing.List[typing.Tuple[int, ...]],
        wr_vs: np.ndarray,
    ):
        """ Handle operation output

        objstr: object
        results: what we will return when we are done with executing operations
        wr_is: write indices
        wr_vs: write values
        """

        if objstr in self.internal_objs:
            # If this is an internal object, just write the values to it
            obj = self.get_internal_object(objstr)
            for (w_i, w_v) in zip(wr_is, wr_vs):
                obj[w_i] = w_v
        else:
            # Otherwise update results
            assert objstr not in results
            results[objstr] = zip(wr_is, wr_vs)

    def execute_ops(self, ops: typing.List[ExecOp]) -> typing.Dict[str, np.ndarray]:
        """ Execute operations """
        if self.xbar_m is None:
            raise RuntimeError("core xbar matrix is undefined")

        execute_ops_debug_ = True
        if ops[0].ty != "MxV":
            print("First operation is not on the crossbar (MxV)")

        # Each operation has a predefined number of inputs, but can have an
        # arbitrary number of outputs, where results are copied. Some outputs,
        # might be used as intermediate results by subsequent operations.
        # These are not returned.
        results = {}
        for op in ops:
            if op.ty == "MxV":
                if len(op.accesses["RD"]) != 1:
                    raise ValueError(
                        "MxV: expecting 1 read argument (got %d)."
                        % (len(op.accesses["RD"],))
                    )
                (rd_objstr, rd_is) = next(iter(op.accesses["RD"].items()))
                if execute_ops_debug_:
                    print("    MxV: RD obj=%s is=%s" % (rd_objstr, rd_is))
                # Fill input vector for mxv
                x = self.read_object(rd_objstr, rd_is)
                y = np.matmul(self.xbar_m, x)
                for (wr_objstr, wr_is) in op.accesses["WR"].items():
                    if execute_ops_debug_:
                        print("    MxV: WR obj=%s is=%s" % (wr_objstr, wr_is))
                    self.handle_op_output(wr_objstr, results, wr_is, y)
            elif op.ty == "ADD":
                if len(op.accesses["RD"]) != 2:
                    raise ValueError(
                        "ADD: expecting 2 read arguments (got %d)."
                        % (len(op.accesses["RD"],))
                    )
                rd_accesses = list(op.accesses["RD"].items())

                (rd_objstr1, rd_is1) = rd_accesses[0]
                obj1_oi = self.objs_info[rd_objstr1]

                (rd_objstr2, rd_is2) = rd_accesses[1]
                obj2_oi = self.objs_info[rd_objstr2]

                # TODO: There are some operations that are required to read the
                # padded object (e.g., CONV) and some (e.g., ADD) that do not.
                # We are curretnly handling this here by passing the
                # appropriate object_info if we want to unpad. However, a
                # better solution would be to just change access relations
                # accordingly. Once we do that, we can remove this here.
                x1 = self.read_object(rd_objstr1, rd_is1, unpad_oi=obj1_oi)
                x2 = self.read_object(rd_objstr2, rd_is2, unpad_oi=obj2_oi)

                if execute_ops_debug_:
                    print("    ADD: RD1 obj=%s is=%s vs=%s" % (rd_objstr1, rd_is1, x1))
                    print("    ADD: RD2 obj=%s is=%s vs=%s" % (rd_objstr2, rd_is2, x2))
                y = np.add(x1, x2)
                for (wr_objstr, wr_is) in op.accesses["WR"].items():
                    if execute_ops_debug_:
                        print("    ADD: WR obj=%s is=%s" % (wr_objstr, wr_is))
                    self.handle_op_output(wr_objstr, results, wr_is, y)
            elif op.ty == "ID":
                if len(op.accesses["RD"]) != 1:
                    raise ValueError(
                        "ID: expecting 2 read arguments (got %d)."
                        % (len(op.accesses["RD"],))
                    )

                ((rd_objstr, rd_is),) = op.accesses["RD"].items()
                x = self.read_object(rd_objstr, rd_is)
                if execute_ops_debug_:
                    print("    ID: RD1 obj=%s is=%s vs=%s" % (rd_objstr, rd_is, x))
                y = x.copy()
                for (wr_objstr, wr_is) in op.accesses["WR"].items():
                    if execute_ops_debug_:
                        print("    ADD: WR obj=%s is=%s" % (wr_objstr, wr_is))
                    self.handle_op_output(wr_objstr, results, wr_is, y)
            else:
                raise ValueError("Unknown operation: %s" % (op.ty,))

        return results

    def write_obj(self, objname: str, w_idx, w_val):
        try:
            self.objs[objname][w_idx] = w_val
        except:
            print("Error accessing object %s on idx=%s" % (objname, w_idx))
            raise

    def validate_write(self, objname: str, w_idx):
        _ = self.objs[objname][w_idx]


class Object:
    """ Object information """

    name: str  # name of the object
    info: ObjectInfo
    reader: typing.Optional[str]  # name of reader stage
    writer: typing.Optional[str]  # name of writer stage

    def __repr__(self):
        return "Object(%s, info=%s)" % (self.name, self.info)

    def __init__(self, name, oi: ObjectInfo):
        self.name = name
        self.info = oi
        self.reader = None
        self.writer = None
        check_class_hints(self)

    def has_reader(self) -> bool:
        return self.reader is not None

    def has_writer(self) -> bool:
        return self.writer is not None

    def set_reader(self, stagename: str):
        """ set reader: we only allow one reader per object """
        if self.reader is not None:
            raise TypeError(
                "failed to set stage %s as reader of object %s because there is already a reader set (stage %s) "
                % (stagename, self.name, self.reader)
            )
        self.reader = stagename

    def set_writer(self, stagename: str):
        """ set writer: we only allow one writer per object """
        if self.writer is not None:
            raise TypeError(
                "failed to set stage %s as writer of object %s because there is already a writer set (stage %s) "
                % (stagename, self.name, self.writer)
            )
        self.writer = stagename

    def is_internal(self) -> bool:
        """ Is this an internal object? i.e., written and read by the same stage """
        return (self.reader is not None) and (self.reader == self.writer)


class PipelineOp:
    """ A pipeline operation described by its input and output buffers """

    po_inps: typing.Dict[str, np.ndarray]
    po_outs: typing.Dict[str, np.ndarray]
    po_outs_done: typing.Set[str]
    po_id: typing.Optional[typing.Any]  # can be set by the user

    def __init__(
        self,
        inps: typing.Dict[str, np.ndarray],
        outs: typing.Dict[str, np.ndarray],
        *,
        completion_fn=None,
        op_id=None,
    ):
        self.po_inps = inps
        self.po_outs = outs
        self.po_outs_done = set()
        self.po_completion_fn = completion_fn
        self.po_id = op_id

    def output_done(self, out: str):
        """ Mark an output finished for this operation """
        assert out in self.po_outs
        assert out not in self.po_outs_done
        self.po_outs_done.add(out)

    def is_completed(self) -> bool:
        return set(self.po_outs.keys()) == self.po_outs_done

    def complete(self):
        if self.po_completion_fn:
            self.po_completion_fn(self)
        else:
            print("OP (id: %s) completed" % (self.po_id,))

class DummyGCU:
    output_objs: typing.Dict[str, np.ndarray]

    def __init__(self):
        """ Initialize the dummy GCU """
        self.output_objs = {}

    def attach_to_pipeline(self, pipeline_write):
        return None

    def tick_gen(self):
        return None

    def get_output_object(self, objname):
        return self.output_objs[objname]

    def init_output_object(self, obj: Object, last_loc: typing.Tuple[int, ...]):
        assert obj.name not in self.output_objs
        self.output_objs[obj.name] = np.zeros(obj.info.get_padded_shape())

    def get_input_wr_a(self, obj: Object) -> typing.Optional[isl.Map]:
        return None

    def write_callback(self, wr_objstr, wr_idx, wr_val):
        """ Write callback for output data

        """
        print("%s: Callback on write: wr_obj:%s wr_idx:%s wr_val:%s"
             % (self.__class__.__name__, wr_objstr, wr_idx, wr_val))
        # if wr_val exsits, and this is an output object, update value
        wr_obj = self.output_objs[wr_objstr]
        assert isinstance(wr_idx, tuple), "wr_idx (%s) not a tuple" % (wr_idx,)
        assert len(wr_obj.shape) == len(wr_idx), "wr_obj.shape (%s) has different dimensions that index (%s)" % (wr_obj.shape, wr_idx,)
        if wr_val is not None:
            wr_obj[wr_idx] = wr_val

    def append_op(self, op: PipelineOp):
        raise TypeError("DummyGCU does not support append_op")

# This is a first implementation of a GCU.
# It manages input and output buffers:
#   - sends data from the input buffers to cores
#   - receives data from the output buffers from cores
#
# Possible additions are:
#  - simulate DMA from/to host memory
#  - send data to cores in batches
#  - have space for multiple buffers on the input cores so that we can transfer
#    full buffers from the GCU. Input buffer on input core will be released
#    when the output buffer is done.
class GCU:
    """ Global control unit """

    po_queue: typing.Deque[PipelineOp]
    po_queue_input_done: typing.Deque[PipelineOp]
    input_objs: typing.Set[str]
    output_objs: typing.Dict[str, np.ndarray]
    output_objs_last_loc: typing.Dict[str, typing.Tuple[int, ...]]
    dummy: bool

    def __init__(self):
        """ Initialize a GCU """
        self.po_queue = deque()
        self.po_queue_input_done = deque()
        self.input_objs = set()
        # TODO: These fields are defined here, but are written by Pipeline
        # code. This is awkward and needs to be fixed.
        self.output_objs = {}
        self.output_objs_last_loc = {}

    def attach_to_pipeline(self, pipeline_write):
        """ We use the pipeline_write method to transfer data from the input
        buffers to the cores """
        self.pipeline_write = pipeline_write

    def init_output_object(self, obj: Object, last_loc: typing.Tuple[int, ...]):
        assert obj.name not in self.output_objs
        self.output_objs[obj.name] = np.zeros(obj.info.get_padded_shape())
        self.output_objs_last_loc[obj.name] = last_loc

    def get_output_object(self, objname):
        return self.output_objs[objname]

    def validate_op(self, op: PipelineOp):
        op_inp_objs = set(op.po_inps.keys())
        op_out_objs = set(op.po_outs.keys())
        gcu_inp_objs = self.input_objs
        gcu_out_objs = set(self.output_objs.keys())
        if op_inp_objs != gcu_inp_objs:
            raise ValueError(
                "op inputs (%s) do not match GCU inputs (%s)"
                % (op_inp_objs, gcu_inp_objs)
            )
        if op_out_objs != gcu_out_objs:
            raise ValueError(
                "op outputs (%s) do not match GCU outputs (%s)"
                % (op_out_objs, gcu_out_objs)
            )

    def append_op(self, op: PipelineOp):
        self.validate_op(op)
        self.po_queue.append(op)

    def get_input_wr_a(self, obj: Object) -> typing.Optional[isl.Map]:
        """ Input object WR accesses relation """
        self.input_objs.add(obj.name)
        shape = obj.info.shape
        idx_names = [
            "%s_%s_i%d" % ("GCU", obj.name, i) for (i,_) in enumerate(shape)
        ]
        obj_names = [
            "%s_i%d" % (obj.name, i) for (i,_) in enumerate(shape)
        ]

        dom = isl_set_from_shape("GCU", idx_names, obj.info.shape)
        rng = isl_set_from_names(obj.name, obj_names)
        rel = isl.Map.from_domain_and_range(dom, rng)

        # NB: For now, the GCU sends input data one-by-one. At some point we
        # might want to implement sending them in batches
        for (idx_n, obj_n) in zip(idx_names, obj_names):
            rel = rel.add_constraint(
                isl.Constraint.eq_from_names(rel.space, {idx_n: 1, obj_n: -1})
            )

        return rel

    def write_callback(self, wr_objstr, wr_idx, wr_val):
        """ Write callback for output data """
        print("%s: Callback on write: wr_obj:%s wr_idx:%s wr_val:%s"
             % (self.__class__.__name__, wr_objstr, wr_idx, wr_val))
        # if wr_val exsits, and this is an output object, update value
        wr_obj = self.output_objs[wr_objstr]
        assert isinstance(wr_idx, tuple), "wr_idx (%s) not a tuple" % (wr_idx,)
        assert len(wr_obj.shape) == len(wr_idx), "wr_obj.shape (%s) has different dimensions that index (%s)" % (wr_obj.shape, wr_idx,)
        if wr_val is not None:
            wr_obj[wr_idx] = wr_val
        last_wr_idx = self.output_objs_last_loc.get(wr_objstr, None)
        if wr_idx == last_wr_idx:
            print("Writing last (%s) index (%s) of object %s" % (last_wr_idx, wr_idx, wr_objstr))
            self.output_done(wr_objstr)

    def output_done(self, objstr: str):
        """ Notification an output for a given operation is finished """
        # NB: This assumes that the output is done *after* the last element of
        # its input has been sent, and that these completions come in order
        # (i.e., all outputs of a single operation will be completed before all
        # outputs of the next operation). There might be cases where this does
        # not hold, so keep this in mind. In that case, we might want to have
        # an operation id to to traverse the pipeline with the commands so that
        # we can track them.
        po = self.po_queue_input_done[0]
        po.output_done(objstr)
        if po.is_completed():
            self.po_queue_input_done.popleft()
            for objstr in self.output_objs:
                np.copyto(po.po_outs[objstr], self.output_objs[objstr])
                self.output_objs[objstr].fill(0)
            po.complete()

    def tick_gen(self):
        """ Tick generator: executes a single tick """
        opid = 0
        while True:
            while len(self.po_queue) == 0:
                print("GCU: Nothing in the queue")
                yield

            curr = self.po_queue.popleft()
            assert curr.po_id is None
            curr.po_id = opid

            inp_its = dict((
                (objname,
                 np.nditer(arr, flags=['multi_index'], op_flags=['readonly']))
                for (objname, arr) in curr.po_inps.items()
            ))

            while True:
                for objname in list(inp_its):
                    it = inp_its[objname]
                    if it.finished:
                        del inp_its[objname]
                        continue
                    self.pipeline_write(self, objname, it.multi_index, it.value.item())
                    it = it.iternext()

                if len(inp_its) == 0:
                    # NB: there is a timing issue here that we need to place
                    # the command to the po_queue_input_done queue and then
                    # yield so that its visible to completions
                    print("GCU: input for op %d done" % (opid,))
                    opid += 1
                    self.po_queue_input_done.append(curr)
                    curr = None
                    yield
                    break
                else:
                    yield

            # TODO: we need to ensure that we do not overwrite useful data
            # here, when we move to another operation. If the objects are big
            # enough, this should not be an issue, but it would be good if we
            # good verify it at the ISL level. Otherwise, we can use multiple
            # buffers.


class Pipeline:
    """ Pipeline """

    p_objs: typing.Dict[str, Object]  # object name -> Object
    p_stages: typing.Dict[str, Stage]  # stage name -> Stage
    p_gcu: typing.Union[GCU,DummyGCU]

    def __init__(
        self,
        stages: typing.List[Stage],
        objs_info: typing.Dict[str, ObjectInfo],
        gcu: GCU = None,
        execute_ops: bool = False,
        loop_inp_limit: typing.Optional[int] = None,
    ):
        """ Initialize a Pipeline

        stages: stages of the pipeline.
        objs_shape: the shapes of the objects accessed in stages.
        gcu: the GCU of the pipeline or None
        execute_ops: actually perform the operations.
        """

        # Initialize p_objs
        self.p_objs = dict((n, Object(n, oi)) for (n, oi) in objs_info.items())

        # Initialize p_stages, and attach stages to the pipeline
        self.p_stages = dict(((s.get_name(), s) for s in stages))
        if len(self.p_stages) != len(stages):
            raise ValueError(
                "stages do not have unique names:\n%s" % (pp.pformat(stages))
            )
        for st in stages:
            st.attach_to_pipeline(self.handle_write, execute_ops)

        # initialize GCU
        self.p_gcu = gcu if gcu is not None else DummyGCU()
        self.p_gcu.attach_to_pipeline(self.handle_write)

        # Discover dependencies and build the loc_to_max_iter relation for
        # every writer/reader pair.
        for st in stages:
            for ro_objname in st.get_ro_objnames():
                if ro_objname not in self.p_objs:
                    raise ValueError(
                        "Object %s read by stage %s, but not provided in initialization"
                        % (ro_objname, st.get_name())
                    )
                obj = self.p_objs[ro_objname]
                obj.set_reader(st.get_name())

            for wo_objname in st.get_wo_objnames():
                if wo_objname not in self.p_objs:
                    raise ValueError(
                        "Object %s written by stage %s, but not provided in initialization"
                        % (wo_objname, st.get_name())
                    )
                obj = self.p_objs[wo_objname]
                obj.set_writer(st.get_name())

            for rw_objname in st.get_rw_objnames():
                if rw_objname not in self.p_objs:
                    raise ValueError(
                        "Object %s is internal to stage %s, but not provided in initialization"
                        % (rw_objname, st.get_name())
                    )
                obj = self.p_objs[rw_objname]
                obj.set_reader(st.get_name())
                obj.set_writer(st.get_name())

        # setup stages and allocate objects based on their dependencies

        # This is a bit awkward, but we do it to allow the pipeline to work
        # without a GCU. In this case, we call set_dont_wait_for_reads() for
        # input objects and set the objects externally.
        for obj in self.p_objs.values():
            if obj.is_internal():
                print("Object %s is internal to %s." % (obj.name, obj.reader))
                reader_stage = self.p_stages[obj.reader]
                reader_stage.core.alloc_internal_object(obj.name, obj.info)

            elif obj.reader is not None:
                reader_stage = self.p_stages[obj.reader]
                reader_stage.core.alloc_object(obj.name, obj.info)

                if obj.writer is not None:
                    # There is a writer, get the write relation from it
                    writer_stage = self.p_stages[obj.writer]
                    wr_a = writer_stage.si.get_obj_wr_rel(obj.name)
                else:
                    # There is no writer, try to get the write relation from
                    # the GCU. Not that this might fail by returning None
                    # (DummyGCU).
                    wr_a = self.p_gcu.get_input_wr_a(obj)

                if wr_a is None:
                    print("Object %s read by %s but written by noone. Assuming it always exists"
                        % (obj.name, obj.reader))
                    reader_stage.set_dont_wait_for_reads(obj.name)
                else:
                    print("Object %s written by %s and read by %s"
                        % (obj, obj.writer if obj.writer is not None else "GCU", obj.reader))
                    rd_a = reader_stage.si.get_obj_rd_rel(obj.name)
                    loc_to_max_iter = isl_rel_loc_to_max_iter(wr_a, rd_a)
                    reader_stage.set_isl_rel_loc_to_max_iter(
                        obj.name, loc_to_max_iter
                    )

            elif obj.writer is not None:
                print("Object %s is an output object: written by %s, but has no readers"
                      % (obj.name, obj.writer))
                writer_stage = self.p_stages[obj.writer]
                last_loc = writer_stage.si.get_obj_last_loc(obj.name, writer_stage.param_vals)
                self.p_gcu.init_output_object(obj, last_loc)
            else:
                print("WARNING: object %s is not read or written" % (obj.name,))

        # Now that we've set loc_to_max_iter, build the python module
        for st in stages:
            st.build_module()

        self.loop_inp_limit = loop_inp_limit
        # Start the generators for every stage
        self.stage_ticks = dict(
            ((s, s.tick_gen(self.loop_inp_limit)) for s in stages)
        )
        # Start the generator for the GCU
        self.gcu_tick = self.p_gcu.tick_gen()
        self.nticks = 0

        self.stages = stages
        # Writes buffer
        self.writes = []

    def configure(self, corecnfs: typing.List[CoreConf]):
        """ Configure the pipeline

        corecnfs: configuration of cores (expected in the same order as stages)
        """
        for (cconf, stage) in zip(corecnfs, self.stages):
            stage.core.configure(cconf)

    def check_writer(self, writer, wr_obj):
        """ Returns true if the writer is supposed to write wr_obj """
        if isinstance(writer, Stage):
            assert writer.get_name() == self.p_objs[wr_obj].writer
        elif isinstance(writer, GCU):
            assert self.p_objs[wr_obj].writer is None
        else:
            assert False, "Unknown writer type: %s (%s)" % (type(writer), writer)

    def handle_write(self, writer, wr_obj, wr_idx, wr_val):
        self.check_writer(writer, wr_obj)
        self.writes.append((wr_obj, wr_idx, wr_val))

    def flush_writes(self):
        """ Actually perform the bufferd writes """
        for (wr_objstr, wr_idx, wr_val) in self.writes:
            reader = self.p_objs[wr_objstr].reader
            if reader is not None:
                # write data on the reader stage
                self.p_stages[reader].write_callback(wr_objstr, wr_idx, wr_val)
            else:
                # write data on GCU
                self.p_gcu.write_callback(wr_objstr, wr_idx, wr_val)

        self.writes = []

    def get_object(self, objstr):
        obj = self.p_objs[objstr]
        reader = obj.reader
        if obj.is_internal():
            stage = self.p_stages[reader]
            return stage.core.get_internal_object(objstr)
        elif reader is not None:
            stage = self.p_stages[reader]
            return stage.core.get_object(objstr)
        else:
            try:
                return self.p_gcu.get_output_object(objstr)
            except KeyError:
                raise ValueError("Could not found object %s" % (objstr,))

    def tick(self):
        s = "TICK: %d" % (self.nticks,)
        # print("="*10, s, "="*(80-10-len(s)-2))

        ret = {}

        stage_keys = list(self.stage_ticks.keys())
        if len(stage_keys) == 0:
            raise StopIteration("No available stages to execute")

        if self.gcu_tick is not None:
            next(self.gcu_tick)

        for s in stage_keys:
            t = self.stage_ticks[s]
            try:
                it = next(t)
                # print("Stage: %s executed iteration %s" % (s.get_name(), it))
            except StopIteration:
                # stage done (typically, because we set a limit)
                print("****** Stage %s done!" % (s.get_name(),))
                del self.stage_ticks[s]
                it = None

            ret[s.get_name()] = it

        print("***** All stages ticked. Flushing writes.")
        self.flush_writes()
        self.nticks += 1
        print("=" * 80)
        return ret

    def tick_gen(self):
        while True:
            try:
                yield self.tick()
            except StopIteration:
                return

    def append_op(self, op: PipelineOp):
        self.p_gcu.append_op(op)
