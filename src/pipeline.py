# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import functools
import itertools
import copy
import typing
import types
import dataclasses as dc
import ast as pyast
import pprint as pp

import numpy as np
import astor as pyastor

from isl_utils import isl_set_to_ast, isl2py_fn, isl_map_to_ast, isl_rel_loc_to_max_iter
from pyast_utils import StructureTupleYields
import islpy as isl

@dc.dataclass(init = False)
class IslAccess:
    """ Wrapper for an isl access mapping: instance space -> object space

    The convention is that the first name of the mapping is the stage, and the
    second is the object (see get_{stage,obj}_name).
    """

    a_ty: str         # RD or WR
    access: isl.Map   # instance space -> object space

    def __init__(self, ty: str, acc: typing.Union[str, isl.Map]):
        if ty != "RD" and ty != "WR":
            raise ValueError("Invalid access type: %s. Expecting 'RD' or 'WR'" % (ty,))

        if isinstance(acc, str):
            try:
                acc = isl.Map(acc)
            except:
                print("Failed to create an isl.Map from %s" % (acc,))
                raise

        if not isinstance(acc, isl.Map):
            raise ValueError("Invalid access type: %s. Expecting str or isl.Map" % (type(acc),))

        self.a_ty = ty
        self.access = acc

    @staticmethod
    def RD(acc: typing.Union[str, isl.Map]) -> 'IslAccess':
        return IslAccess("RD", acc)

    @staticmethod
    def WR(acc: typing.Union[str, isl.Map]) -> 'IslAccess':
        return IslAccess("WR", acc)

    def get_stage_name(self) -> str:
        return self.access.get_tuple_name(isl.dim_type.in_)

    def get_obj_name(self) -> str:
        return self.access.get_tuple_name(isl.dim_type.out)

    def get_idx_ndims(self) -> int:
        return self.access.space.dim(isl.dim_type.in_)

    def get_obj_ndims(self) -> int:
        return self.access.space.dim(isl.dim_type.out)

@dc.dataclass(init = False)
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
    unconnected input is an ojbect that is read, but not written within this
    stage, and vice-versa for unconnected outputs). These are the read and
    write access relations for a stage.
    """
    ops: typing.List[OpInfo]
    ro_objs: typing.Set[str] # Objects that this stage reads (external)
    wo_objs: typing.Set[str] # Objects that this stage writes (external)
    rw_objs: typing.Set[str] # Objects that this stage writes and reads (internal)

    def __init__(self, ops: typing.List[OpInfo]):
        if ops[0].op_ty != 'MxV':
            raise ValueError("First operation should be MxV (instead is:%s)" % (ops[0].op_ty))
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
                    if objname in self.wo_objs: # Object was previously written, no read. Move it to rw
                        self.wo_objs.remove(objname)
                        self.rw_objs.add(objname)
                    elif objname in self.ro_objs: # Object was previously read, do nothing
                        pass
                    elif objname in self.rw_objs: # Object was previously written and read, do nothing
                        pass
                    else: # First time seeing this object, put it into ro set
                        self.ro_objs.add(objname)
                elif acc.a_ty == "WR":
                    if objname in self.wo_objs:
                        # Object was previously written, error
                        raise ValueError("Object %s written in op %s (id:%d) but also written previously" % (objname, op.op_ty, op_id))
                    elif objname in self.ro_objs:
                        raise ValueError("Object %s written in op %s (id:%d) previously read" % (objname, op.op_ty, op_id))
                    elif objname in self.ro_objs:
                        raise ValueError("Object %s written in op %s (id:%d) previously write and read" % (objname, op.op_ty, op_id))
                    else:
                        self.wo_objs.add(objname)
                else: ValueError("Unknown access type: %s" % (acc.a_ty,))

    def get_stage_name(self) -> str:
        return self.ops[0].get_stage_name()

    def get_domain(self) -> isl.Map:
        return self.ops[0].get_domain()

    def get_obj_rd_rel(self, objname: str) -> isl.Map:
        if objname not in self.ro_objs:
            raise ValueError("stage %s does not read from object %s"  % (self.get_stage_name(), objname))
        accs = []
        for op in self.ops:
            accs.extend(a for a in op.rd_accesses() if a.get_obj_name() == objname)
        if len(accs) > 1:
            raise NotImplementedError("I guess we need to combine multiple read accesses...")
        return accs[0].access

    def get_obj_wr_rel(self, objname: str) -> isl.Map:
        if objname not in self.wo_objs:
            raise ValueError("stage %s does not write to object %s"  % (self.get_stage_name(), objname))
        accs = []
        for op in self.ops:
            accs.extend(a for a in op.wr_accesses() if a.get_obj_name() == objname)
        assert len(accs) == 1, "Multiple writes on the same object not allowed"

        return accs[0].access

def isl_map_to_pyfn(rel, fnname, s = None):
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
        s = (nin, nout) = tuple(rel.space.dim(x) for x in (isl.dim_type.in_, isl.dim_type.out))
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
        except(ValueError):
            print("%s: ri=%s cannot be packed into (idx,loc)" % (rel_iter.__name__, ri,))
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

@dc.dataclass(init = False)
class ExecOp:
    ty: str
    accesses: typing.Dict[str, typing.Dict[str, typing.Optional[typing.List[typing.Tuple[int, ...] ] ] ] ]

    def __init__(self, ty, rd_objs, wr_objs):
        self.ty = ty
        self.accesses = {}
        self.accesses['RD'] = dict( (o, None) for o in rd_objs)
        self.accesses['WR'] = dict( (o, None) for o in wr_objs)

class AccessIterator:
    idx: typing.Tuple[int, ...]
    ops: typing.List[ExecOp]

    def __init__(self, stage):
        self.idx_ = None
        self.ops = [
            ExecOp(ty=op.op_ty, rd_objs=op.rd_objs(), wr_objs=op.wr_objs())
            for op in stage.si.ops
        ]
        self.stage_name = stage.get_name() # For debugging messages
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
                    assert self.idx_ == idx, "Expecting idx=%s but got idx=%s" % (self.idx_, idx)
                self.set_access(op_id, a_ty, obj, access_l)
                yield

        def decorator(fn):
            wrapped_fn = lambda : update_state_dec(rel_a_iter(fn))
            self.fns.append(wrapped_fn)
            return fn
        return decorator

    def loop(self) -> typing.Iterator[typing.List[ExecOp]]:
        for inp in itertools.count():
            iters = [ f() for f in self.fns ]
            niters = 0
            for _ in zip(*iters):
                # print("%s: LOOP => %s " % (self.stage_name, self.idx_))
                niters += 1
                yield ((inp,) + self.idx_, self.ops)
                self.reset_access()
            assert(niters > 0)

class LocToMaxIterIterator:

    def __init__(self, stage):
        # Maximum iteration allowed for every object that needs to be read
        #  if maximum iteration is None, no writes have happened yet.
        self.obj_max_iter = dict((o, None) for o in stage.si.ro_objs)

        # Initialized generators
        self.max_iter_gs = {}

        self.stage_name = stage.get_name() # debugging

    def register_iter_fn(self, obj):
        def max_iter_gen(fn):
            """ Python generator that consumers writes and updates self.obj_max_iter """

            print("%s: Initializing max_iter_gen generator for object: %s (%s)" % (self.stage_name, obj, fn))
            # inp  is the id of the input (typically image) being processed
            # This is used to maintain proper ordering when we wrap-around
            for inp in itertools.count():
                for (write, max_iter) in fn():
                    while True:
                        new_write = yield
                        if new_write == write:
                            self.obj_max_iter[obj] = (inp,) + max_iter
                            print("%s:\tGot expected write: %s. max_iter is now: %s" % (self.stage_name, new_write, self.obj_max_iter[obj]))
                            break
                        else:
                            print("%s:\tGot %s, but expecting %s to change max_iter" % (self.stage_name, new_write, write))

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
                print("%s: max_iter for object %s unset. Reads not ready" % (self.stage_name, obj))
                return False
            assert isinstance(max_iter, tuple)
            assert len(idx) == len(max_iter)
            return idx <= max_iter

        # NB: if the iteratable is empty, all() returns true
        return all(obj_rdy(idx, o, max_i) for (o, max_i) in self.obj_max_iter.items())

class Stage:
    def __init__(self, si: StageInfo, param_vals = None):
        """ Initialize a stage

        si: StageInfo for this stage
        param_vals: dict of values for parameters in SI experessions (or None).

        The @param_vals argument is passed to the execution of the generated
        python modules.
        """
        self.si = si
        self.param_vals = param_vals if param_vals is not None else dict()
        self.print_ast_ = True

        # For every object this stage needs to read, this dict stores the ISL
        # relation that provides the following mapping:
        #  observed object writes to the maximum iteration that can be executed.
        # This relation is set by the pipeline
        self.loctomaxiter_rel = dict((o, None) for o in self.si.ro_objs)

        # Helpers for combining generated iterators
        #  They provide decorators that are added in the generated code
        self.access_i = AccessIterator(self)
        self.loctomaxiter_i = LocToMaxIterIterator(self)

    def get_rd_objnames(self) -> typing.Set[str]:
        """ Return the objects that this stage reads """
        return self.si.ro_objs

    def get_wr_objnames(self) -> typing.Set[str]:
        """ Return the objects that this stage reads """
        return self.si.wo_objs

    def attach_to_pipeline(self, pipeline_write, execute_ops):
        self.pipeline_write = pipeline_write
        self.core = Core()
        self.execute_ops = execute_ops

    def set_isl_rel_loc_to_max_iter(self, objname: str, rel: isl.Map):
        """ set the relation to compute the maximum iteration based on writes """
        if self.loctomaxiter_rel[objname] is not None:
            raise ValueError("loctomaxiter_rel alredy set for object %s" % (objname,))
        self.loctomaxiter_rel[objname] = rel

    def set_dont_wait_for_reads(self, objname):
        """ Remove objname from obj_max_iter so that we never wait for reads on this object """
        if self.loctomaxiter_rel[objname] is not None:
            raise ValueError("loctomaxiter_rel alredy set for object %s" % (objname,))
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
                fn_name = "%s_%02d_%s_%s_%s" % (self.get_name(), op_id, op.op_ty, acc.a_ty.lower(), acc.get_obj_name())
                py = isl_map_to_pyfn(acc.access, fn_name)
                s = "access_iter(op_id=%d, op_ty='%s', a_ty='%s', obj='%s')" \
                  % (op_id, op.op_ty, acc.a_ty, acc.get_obj_name())
                dec_call = pyast.parse(s).body[0].value
                py.decorator_list.append(dec_call)
                body.append(py)

        # generate loc_to_max_iter() functions for every object read by this stage
        for (obj,rel) in self.loctomaxiter_rel.items():
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
            print("-"*10, s, "-"*(80-10-len(s)-2))
            # print(astpp_dump(ast_mod))
            # print("-"*80)
            print(pyastor.to_source(ast_mod))
            print("-"*80)


        pyast.fix_missing_locations(ast_mod)
        code = compile(ast_mod, "<generated>", "exec")
        ret = types.ModuleType("stage_%s" % (self.get_name()))
        ret.__dict__.update(self.param_vals)
        ret.__dict__.update({'access_iter': self.access_i.register_iter_fn})
        ret.__dict__.update({'loc_to_maxiter_iter': self.loctomaxiter_i.register_iter_fn})
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

        Execute the "snooping for SRAM writes" logic.
        """

        print("%s: Callback on write: wr_obj:%s wr_idx:%s wr_val:%s" % (self.get_name(), wr_objstr, wr_idx, wr_val))

        # the write should be in the object that we read
        assert wr_objstr in self.si.ro_objs

        # Write data (if a value exists)
        if wr_val is not None:
            self.core.write_obj(wr_objstr, wr_idx, wr_val)
        else:
            self.core.validate_write(wr_objstr, wr_idx)

        self.loctomaxiter_i.handle_write(wr_objstr, wr_idx)

    def tick_gen(self):
        """ Tick generator: executes a single tick

        yields the iteration executed or None if it was stalled
        """
        for (idx, ops) in self.access_i.loop():
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
    width: int = 256 #
    xbar_m: typing.Optional[np.ndarray]
    # NB: For now, we just keep objects as np arrays. Eventually, we might want
    # to map them to a linear buffer representing the core's SRAM.
    objs: typing.Dict[str,np.ndarray]

    def __init__(self):
        self.xbar_m = None
        self.objs = {}

    def configure(self, cnf: CoreConf):
        self.set_xbar_matrix(cnf.xbar_m)

    def set_xbar_matrix(self, xbar_m):
        (xbar_m_width, xbar_m_height) = xbar_m.shape
        # we accept whatever matrix we are given, as long as it fits into the
        # crossbar.
        if xbar_m_width > self.width or xbar_m_height > self.width:
            raise ValueError("XBAR too small: XBAR width is %d, while given matrix shape is:%s" % (self.width, xbar_m.shape))
        self.xbar_m = xbar_m.copy()

    def alloc_object(self, objname: str, shape: typing.Tuple[int, ...]):
        obj = np.zeros(shape)
        self.set_object(objname, obj)

    def set_object(self, objname: str, obj: np.ndarray):
        if objname in self.objs:
            raise ValueError("object %s already exists" % (objname,))
        self.objs[objname] = obj

    def get_object(self, objname: str):
        return self.objs[objname]

    def validate_ops(self, ops: ExecOp):
        """ Valdidate operations

        Ensure that the reads are within the array bounds.
        Will raise an error if that's not the case
        """

        # TODO: We need to handle the case where one op reads the output of
        # another op. In this case, we should read the data from @ret, and
        # delete them so that they are not part of the returned result.
        ret = {}
        for op in ops:
            for (rd_objstr, rd_is) in op.accesses["RD"].items():
                if rd_objstr not in self.objs:
                    raise ValueError("object %s does not exist in this core" % (rd_objstr,))
                obj = self.objs[rd_objstr]
                for idx in rd_is:
                    assert isinstance(idx, tuple) and len(idx) == len(obj.shape), "idx=%s obj.shape=%s" % (idx, obj.shape)
                    try:
                        _ = obj[idx]
                    except:
                        print("Failed to access %s (shape=%s) on %s" % (objstr, obj.shape, idx))
                        raise
                for (wr_objstr, wr_is) in op.accesses["WR"].items():
                    assert wr_objstr not in ret
                    ret[wr_objstr] = zip(wr_is, itertools.repeat(None))

        return ret

    def read_object(self, objstr, rd_is):
        if objstr not in self.objs:
            raise ValueError("object %s does not exist in this core" % (objstr,))
        obj = self.objs[objstr]
        # NB: not sure if we need to deal with multi-dimensional objects or how
        # to. For now assume, that objects stored on cores are 1D
        ret = np.zeros(shape=(len(rd_is),) )
        for i,idx in enumerate(rd_is):
            assert isinstance(idx, tuple) and len(idx) == len(obj.shape), "idx=%s obj.shape=%s" % (idx, obj.shape)
            ret[i] = obj[idx]
        return ret

    def execute_ops(self, ops: ExecOp) -> typing.Dict[str, np.ndarray]:
        """ Execute operations """
        if self.xbar_m is None:
            raise RuntimeError("core xbar matrix is undefined")

        assert ops[0].ty == "MxV", "First operation should be on the crossbar (MxV)"

        # TODO: We need to handle the case where one op reads the output of
        # another op. In this case, we should read the data from @ret, and
        # delete them so that they are not part of the returned result.
        ret = {}
        for op in ops:
            if op.ty == "MxV":
                if len(op.accesses["RD"]) != 1:
                    raise ValueError("MxV: expecting 1 read argument (got %d)." % (len(op.accesses['RD'], )))
                (rd_objstr, rd_is) = next(iter(op.accesses["RD"].items()))
                if rd_objstr not in self.objs:
                    raise ValueError("object %s does not exist in this core" % (rd_objstr,))
                rd_obj = self.objs[rd_objstr]
                # Fill-in input vector for mxv
                x = self.read_object(rd_objstr, rd_is)
                y = np.matmul(self.xbar_m, x)
                # For every output object, zip the result with the indices 
                for (wr_objstr, wr_is) in op.accesses["WR"].items():
                    assert wr_objstr not in ret
                    ret[wr_objstr] = zip(wr_is, y)
            elif  op.ty == "ADD":
                if len(op.accesses["RD"]) != 2:
                    raise ValueError("ADD: expecting 2 read arguments (got %d)." % (len(op.accesses['RD'], )))
                rd_accesses = list(op.accesses["RD"].items())
                (rd_objstr1, rd_is1) = rd_accesses[0]
                x1 = self.read_object(rd_objstr1, rd_is1)
                (rd_objstr2, rd_is2) = rd_accesses[1]
                x2 = self.read_object(rd_objstr2, rd_is2)
                y = np.add(x1, x2)
                for (wr_objstr, wr_is) in op.accesses["WR"].items():
                    assert wr_objstr not in ret
                    ret[wr_objstr] = zip(wr_is, y)
            else:
                raise ValueError("Unknown operation: %s" % (op.ty,))

        return ret


    def write_obj(self, objname: str, w_idx, w_val):
        self.objs[objname][w_idx] = w_val

    def validate_write(self, objname: str, w_idx):
        _= self.objs[objname][w_idx]

class Object:
    name: str # name of the object
    shape: typing.Tuple[int,...] # shape of the object
    reader: typing.Optional[str] # name of reader stage
    writer: typing.Optional[str] # name of writer stage

    def __repr__(self):
        return "Object(%s, shape=%s)" % (self.name, self.shape)

    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.reader = None
        self.writer = None

    def has_reader(self) -> bool:
        return self.reader is not None

    def has_writer(self) -> bool:
        return self.writer is not None

    def set_reader(self, stagename: str):
        """ set reader: we only allow one reader per object """
        if self.reader is not None:
            raise TypeError("failed to set stage %s as reader of object %s because there is already a reader set (stage %s) " % (stagename, self.name, self.reader))
        self.reader = stagename

    def set_writer(self, stagename: str):
        """ set writer: we only allow one writer per object """
        if self.writer is not None:
            raise TypeError("failed to set stage %s as writer of object %s because there is already a writer set (stage %s) " % (stagename, self.name, self.writer))
        self.writer = stagename

class Pipeline:
    """ Pipeline """

    p_objs:   typing.Dict[str, Object] # object name -> Object
    p_stages: typing.Dict[str, Stage]  # stage name -> Stage

    # Orphan objects are objects without a reader, and they are kept here
    orphan_objs: typing.Dict[str, np.ndarray]

    def __init__(self,
                 stages: typing.List[Stage],
                 objs_shape: typing.Dict[str, typing.Tuple[int, ...]],
                 execute_ops: bool = False):
        """ Initialize a Pipeline

        stages: stages of the pipeline.
        objs_shape: the shapes of the objects accessed in stages.
        execute_ops: actually perform the operations.
        """

        # Initialize p_objs
        self.p_objs = dict((n, Object(n,s)) for (n,s) in objs_shape.items())

        # Initialize p_stages, and attach stages to the pipeline
        self.p_stages = dict( ((s.get_name(), s) for s in stages) )
        if len(self.p_stages) != len(stages):
            raise ValueError("stages do not have unique names:\n%s" % (pp.pformat(stages)))
        for st in stages:
            st.attach_to_pipeline(self.handle_write, execute_ops)

        # Discover dependencies and build the loc_to_max_iter relation for
        # every writer/reader pair.
        for st in stages:
            for rd_objname in st.get_rd_objnames():
                if rd_objname not in self.p_objs:
                    raise ValueError("Object %s read by stage %s, but not provided in initialization" % (rd_objname, st.get_name()))
                obj = self.p_objs[rd_objname]
                obj.set_reader(st.get_name())

            for wr_objname in st.get_wr_objnames():
                if wr_objname not in self.p_objs:
                    raise ValueError("Object %s written by stage %s, but not provided in initialization" % (wr_objname, st.get_name()))
                obj = self.p_objs[wr_objname]
                obj.set_writer(st.get_name())

        # setup stages and allocate objects based on their dependencies
        self.orphan_objs = {}
        for obj in self.p_objs.values():
            if obj.reader is not None:
                reader_stage = self.p_stages[obj.reader]
                reader_stage.core.alloc_object(obj.name, obj.shape)
                if obj.writer is None:
                    print("Object %s read by %s has no writer. Assuming it always exists." % (obj.name, obj.reader))
                    reader_stage.set_dont_wait_for_reads(obj.name)
                else:
                    print("Object %s written by %s and read by %s" % (obj, obj.writer, obj.reader))
                    writer_stage = self.p_stages[obj.writer]
                    rd_a = reader_stage.si.get_obj_rd_rel(obj.name)
                    wr_a = writer_stage.si.get_obj_wr_rel(obj.name)
                    loc_to_max_iter = isl_rel_loc_to_max_iter(wr_a, rd_a)
                    reader_stage.set_isl_rel_loc_to_max_iter(obj.name, loc_to_max_iter)

            elif obj.writer is not None:
                print("Object %s is orphan: written by %s, but has no readers" % (obj.name, obj.writer))
                self.orphan_objs[obj.name] = np.zeros(obj.shape)
            else:
                print("WARNING: object %s is not read or written" % (obj.name,))


        # Now that we've set loc_to_max_iter, build the python module
        for st in stages:
            st.build_module()

        # Start the generators for every stage
        self.stage_ticks = [ (s,s.tick_gen()) for s in stages ]
        self.nticks = 0

        self.stages = stages
        # Writes buffer
        self.writes = []

    def configure(self, corecnfs: typing.List[CoreConf]):
        """ Configure the pipeline

        corecnfs: configuration of cores (expected in the same order as stages)
        """
        for (cconf,stage) in zip(corecnfs, self.stages):
            stage.core.configure(cconf)

    def handle_write(self, stage, wr_obj, wr_idx, wr_val):
        assert stage.get_name() == self.p_objs[wr_obj].writer
        self.writes.append((wr_obj, wr_idx, wr_val))

    def flush_writes(self):
        for (wr_objstr, wr_idx, wr_val) in self.writes:
            reader = self.p_objs[wr_objstr].reader
            if reader is not None:
                # execute callback on the reader
                self.p_stages[reader].write_callback(wr_objstr, wr_idx, wr_val)
            elif wr_val is not None and wr_objstr in self.orphan_objs:
                # if wr_val exsits, and this is an orphan object, update value
                wr_obj = self.orphan_objs[wr_objstr]
                assert isinstance(wr_idx, tuple) and len(wr_obj.shape) == len(wr_idx)
                wr_obj[wr_idx] = wr_val

        self.writes = []

    def get_object(self, objstr):
        reader = self.p_objs[objstr].reader
        if reader is not None:
            stage = self.p_stages[reader]
            return stage.core.get_object(objstr)
        elif objstr in self.orphan_objs:
            return self.orphan_objs[objstr]
        else:
            return None

    def tick(self):
        s = "TICK: %d" % (self.nticks,)
        print("="*10, s, "="*(80-10-len(s)-2))
        ret = {}
        for (s,t) in self.stage_ticks:
            it = next(t)
            ret[s.get_name()] = it
            #print("Stage: %s executed iteration %s" % (s.get_name(), it))

        print("***** All stages ticked. Flushing writes.")
        self.flush_writes()
        self.nticks += 1
        print("="*80)
        return ret
