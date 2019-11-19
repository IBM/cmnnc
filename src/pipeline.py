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
import ast as pyast

import numpy as np
import astor as pyastor

from isl_utils import isl_set_to_ast, isl2py_fn, isl_map_to_ast, isl_rel_loc_to_max_iter
import islpy as isl

class StageInfo:
    """ Polyhedral information for a stage """
    rd_a: isl.Map # Read access relation
    wr_a: isl.Map # Write access relation
    id_s: isl.Set # Index space (i.e., Instance set)

    def __init__(self, rd_a=None, wr_a=None):
        if isinstance(rd_a, str):
            rd_a = isl.Map(rd_a)
        if isinstance(wr_a, str):
            wr_a = isl.Map(wr_a)

        if rd_a is None and wr_a is None:
            raise ValueError("rd_a and wr_a cannot both be None")

        # Check that the domain of the relations matches the index space
        # Note that:
        #    >>> isl.Set("{S[i]: 0 <= i < 10}") == isl.Map("{ S[i] -> in1[i] : 0 <= i < 10 }").domain()
        #    True
        #    >>> isl.Set("{S[i]: 0 <= i < 10}") == isl.Map("{ X[i] -> in1[i] : 0 <= i < 10 }").domain()
        #    False
        # I.e., the identifiers need to match
        #
        # NB: Turns out, this is not always trivial to do, so we disable the
        # check for now We do check, that the read and write iterator maps
        # produce the same first part, so we should be OK.
        #
        # if (rd_a is not None) and (wr_a is not None):
        #     assert rd_a.domain() == wr_a.domain(), "\n rd_a: %s\n wr_a:%s\n" % (rd_a.domain(), wr_a.domain())

        self.id_s = rd_a.domain() if rd_a is not None else wr_a.domain()
        self.rd_a = rd_a
        self.wr_a = wr_a

    def __repr__(self):
        return "StageInfo(rd_a=%s, wr_a=%s)" % (self.rd_a, self.wr_a)

    def get_wr_obj_name(self):
        # Assuming a single object for now
        return self.wr_a.get_tuple_name(isl.dim_type.out) if self.wr_a is not None else None

    def get_rd_obj_name(self):
        # Assuming a single object for now
        return self.rd_a.get_tuple_name(isl.dim_type.out) if self.rd_a is not None else None

    def get_name(self):
        # Not sure how reiliable this is, but it works for now...
        return self.id_s.get_tuple_name()

class StructureTupleYields(pyast.NodeTransformer):
    """ AST transformer for "structuring" yielded tuples

        For example, if structure is (2,3), then a yield expression, yielding a
        5-tuple: yield (a,b,c,d,e) will be transformed to yield ((a,b,),(c,d,e)).
    """
    def __init__(self, structure):
        super().__init__()
        self.structure = structure

    def visit_Yield(self, node):
        # This yield is not a tuple, do nothing
        if not isinstance(node.value, pyast.Tuple):
            print("*"*10, "Yiedling something which is not a tuple. Doing nothing")
            return node

        elts = node.value.elts
        ctx = node.value.ctx
        nelts = len(elts)
        if nelts != sum(self.structure):
            print("*"*10, "Yiedling a tuple with size=%d while structure=%s. Doing nothing." % (nelts, structure))
            return node

        new_elts = []
        elts_iter = iter(elts)
        for n in self.structure:
            xelts = [ x for x in itertools.islice(elts_iter, n) ]
            xtuple = pyast.Tuple(xelts, copy.copy(ctx))
            new_elts.append(xtuple)

        # sanity check that there are no more elements in the iterator
        # (they shouldn't be since we checked the length)
        try:
            next(elts_iter)
            assert False
        except StopIteration:
            pass

        new_node = pyast.Yield(pyast.Tuple(new_elts, copy.copy(ctx)))
        return pyast.copy_location(new_node, node)

def structure_yields_example():
    rel = isl.Map("{ A[i,j] -> B[k] : 0 <= i < 10 and 0 <= j < 3 and k = i + j }")
    rel_ast = isl_map_to_ast(rel)
    structure = (ndims_in, ndims_out) =  tuple(rel.space.dim(x) for x in (isl.dim_type.in_, isl.dim_type.out))
    py = isl2py_fn(rel_ast, "foo")
    print("Before:\n", pyastor.to_source(py))
    StructureTupleYields(structure).visit(py)
    print("After:\n", pyastor.to_source(py))

def isl_map_to_pyfn(rel, fnname):
    """ Transform an isl map to a python function """
    ast = isl_map_to_ast(rel)
    py = isl2py_fn(ast, fnname)
    # isl_map_to_ast works by generating code for rel.unwrap() The resulting
    # python function returns a tuple containing items from the doimain and the
    # image of the relationship, but there is no way to distinguish which is
    # which.
    #
    # Here, we apply a transformation that structures the yields of the
    # generated function so that they yield a 2-tuple of tuples, one for the
    # doimain and one for the image.
    s = (nin, nout) = tuple(rel.space.dim(x) for x in (isl.dim_type.in_, isl.dim_type.out))
    StructureTupleYields(s).visit(py)
    return py

class IterInf(object):
    pass

class Stage:
    def __init__(self, si, param_vals = None):
        """ Initialize a stage

        si: StageInfo for this stage
        param_vals: dict of values for parameters in SI experessions (or None)
        execute_ops: execute operations
        """
        self.si = si
        self.param_vals = param_vals if param_vals is not None else dict()
        self.print_ast_ = True
        self.isl_rel_loc_to_max_iter = None

        self.max_iter = None   # Current max iteration
        self.max_iter_g = None # Generator that consumes writes to objects and updates self.max_iter

        # These are just used for debuging.
        # For now, we only support a single read and a single write object.
        self.rd_obj = None
        self.wr_obj = None

        # NB: set in pipeline instantiation
        self.pipeline_write = None
        self.core = None
        self.execute_ops = False

    def __repr__(self):
        return "Stage(StageInfo(id_s=%s, rd_a=%s, wr_a=%s))" % (self.si.id_s, self.si.rd_a, self.si.wr_a)

    def set_rd_obj(self, obj):
        assert self.rd_obj is None, "Only one read object is allowed (FIXME)"
        self.rd_obj = obj

    def set_wr_obj(self, obj):
        assert self.wr_obj is None, "Only one read object is allowed (FIXME)"
        self.wr_obj = obj

    def set_isl_rel_loc_to_max_iter(self, rel):
        """ set the relation to compute the maximum iteration based on writes """
        assert self.si.rd_a is not None
        self.isl_rel_loc_to_max_iter = rel

    def set_dont_wait_for_reads(self):
        """ Set the maximum iteration to an "infinite" value, so that we never
        wait for writes """
        assert self.si.rd_a is not None
        self.isl_rel_loc_to_max_iter is None, "set_isl_rel_loc_to_max_iter() was already called"
        self.max_iter = IterInf()

    def build_module(self):
        self.pymod = self.build_module_()

    def build_module_(self):
        body = []

        if True:
            ast = isl_set_to_ast(self.si.id_s)
            py = isl2py_fn(ast, "idx_iter")
            body.append(py)

        if self.si.rd_a is not None:
            py = isl_map_to_pyfn(self.si.rd_a, "rd_a_iter")
            body.append(py)

        if self.si.wr_a is not None:
            py = isl_map_to_pyfn(self.si.wr_a, "wr_a_iter")
            body.append(py)


        if self.isl_rel_loc_to_max_iter is not None:
            py = isl_map_to_pyfn(self.isl_rel_loc_to_max_iter, "loc_to_max_iter")
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
        exec(code, ret.__dict__)
        return ret

    def get_name(self):
        return self.si.get_name()

    def get_wr_obj_name(self):
        return self.si.get_wr_obj_name()

    def get_rd_obj_name(self):
        return self.si.get_rd_obj_name()

    def rel_a_iter(self, rel_iter):

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


    def loop_it(self):
        """ Loop generator: iterates over indexes, reads, and writes """

        # None loop, used when we do not have readers or writers
        def none_loop():
            for i in self.pymod.idx_iter():
                yield (i, None)

        # inp  is the id of the input (typically image) being processed
        # This is used to maintain proper ordering when we wrap-around
        for inp in itertools.count():
            # Create two iterators, one of reads and one for writes.
            # Both return tuples with the first element being the loop index, and
            # the second a list of read or write accesses.
            iter_rd = self.rel_a_iter(self.pymod.rd_a_iter) if self.si.rd_a is not None else none_loop()
            iter_wr = self.rel_a_iter(self.pymod.wr_a_iter) if self.si.wr_a is not None else none_loop()
            for ((rd_i, rd), (wr_i, wr)) in zip(iter_rd, iter_wr):
                assert rd_i == wr_i, "%s: rd_i=%s does not match wr_i=%s" % (self.get_name(), rd_i, wr_i)
                i = (inp,) + rd_i
                # print("%s:" % (self.get_name()), "i:", i, "rd:", rd, "wr:", wr)
                yield (i, rd, wr)

    def reads_ready(self, i):
        """ Return whether the reads for iteration i are ready """
        if self.si.rd_a is None:
            # No read access specified, reads are always ready
            return True

        if self.max_iter is None:
            print("%s: max_iter unset. Reads not ready" % self.get_name())
            return False

        if isinstance(self.max_iter, IterInf):
            print("%s: max_iter set to infinite. All reads ready." % self.get_name())
            return True

        # Sanity check that iteration i has the same dimensions with max_iter
        assert len(self.max_iter) == len(i)

        ret = i <= self.max_iter
        # print("%s: max_iter:%s i:%s reads_ready=%s" % (self.get_name(), self.max_iter, i, ret))
        return ret

    def issue_write(self, wr_obj, wr_idx, wr_val):
        # print("%s: Issuing write: obj:%s idx:%s" % (self.get_name(), wr_obj, wr_idx, wr_val))
        if self.pipeline_write is not None:
            self.pipeline_write(wr_obj, wr_idx, wr_val)

    def print_loc_to_max_iter(self):
        print("-------> locl_to_max_iter rel for %s" % (self.get_name(),))
        for (write, max_i) in self.pymod.loc_to_max_iter():
            print("w:%20s => max_iter:%s" % (write, max_i))
        print("<---------------")

    def print_rd_a_iter(self):
        print("-------> print_rd_a_iter for %s" % (self.get_name(),))
        for (idx, rd_a) in self.pymod.rd_a_iter():
            print("idx:%s => rd:%s" % (idx, rd_a))
        print("<---------------")

    def max_iter_gen(self):
        """ Python generator that consumers writes and updates self.max_iter """

        # inp  is the id of the input (typically image) being processed
        # This is used to maintain proper ordering when we wrap-around
        for inp in itertools.count():
            for (write, max_iter) in self.pymod.loc_to_max_iter():
                while True:
                    new_write = yield
                    if new_write == write:
                        self.max_iter = (inp,) + max_iter
                        print("%s:\tGot expected write: %s. max_iter is now: %s" % (self.get_name(), new_write, self.max_iter))
                        break
                    else:
                        print("%s:\tGot %s, but expecting %s to change max_iter" % (self.get_name(), new_write, write))

    def write_callback(self, wr_obj, wr_idx, wr_val):
        """ Write callback executed on the reader

        Execute the "snooping for SRAM writes" logic.
        """

        print("%s: Callback on write: wr_obj:%s wr_idx:%s wr_val:%s" % (self.get_name(), wr_obj, wr_idx, wr_val))

        # the write should be in the object that we read
        assert self.get_rd_obj_name() == wr_obj

        # Write data (if a value exists)
        if wr_val is not None:
            self.core.write_obj(wr_obj, wr_idx, wr_val)
        else:
            self.core.validate_accesses(wr_obj, [wr_idx])

        # Initialize the max iteration generator, if it does not exist
        if self.max_iter_g is None:
            self.max_iter_g = self.max_iter_gen()
            self.max_iter_g.send(None)

        # Call the generator that consumes writes
        self.max_iter_g.send(wr_idx)

    def tick_gen(self):
        """ Tick generator: executes a single tick

        yields the iteration executed
        """
        for (i, rd_is, wr_is) in self.loop_it():
            while not self.reads_ready(i):
                rds = "%s%s" % (self.rd_obj, rd_is) if rd_is is not None else None
                print("%s: Stalling iteration %s. Reads (%s) not ready." % (self.get_name(), i, rds))
                yield i
            rds = "%s%s" % (self.rd_obj, rd_is) if rd_is is not None else None
            wrs = "%s%s" % (self.wr_obj, wr_is) if wr_is is not None else None
            print("%s: Executing iteration %s.\n RDs: %s\n WRs: %s" % (self.get_name(), i, rds, wrs))
            del rds, wrs

            if self.execute_ops:
                out = self.core.execute_ops(self.rd_obj, rd_is if rd_is is not None else [])
                assert wr_is is None or out.size == len(wr_is),"output size: %d (shape=%s) does not match number of indices: %d" % (out.size, out.shape, len(wr_is))
                wr_vs = iter(out)
            else:
                self.core.validate_accesses(self.rd_obj, rd_is if rd_is is not None else [])
                wr_vs = itertools.repeat(None, len(wr_is)) if wr_is is not None else []

            if wr_is is not None:
                for (wr_i,wr_v) in zip(wr_is, wr_vs):
                    self.issue_write(self.get_wr_obj_name(), wr_i, wr_v)
            yield i

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

    def validate_accesses(self, objstr: str, idxs: typing.List[typing.Tuple[int,...]]):
        """ Valdidate operations

        Ensure that the reads are within the array bounds.
        Will raise an error if that's not the case
        """
        # Perform the read and see if we get any out-of-bounds error
        if objstr not in self.objs:
            raise ValueError("object %s does not exist in this core" % (objstr,))
        obj = self.objs[objstr]
        for idx in idxs:
            assert isinstance(idx, tuple) and len(idx) == len(obj.shape), "idx=%s obj.shape=%s" % (idx, obj.shape)
            try:
                _ = obj[idx]
            except:
                print("Failed to access %s (shape=%s) on %s" % (objstr, obj.shape, idx))
                raise

    def execute_ops(self, rd_objstr: str, rd_is: typing.List[typing.Tuple[int,...]]) -> np.ndarray:
        """ execute operations

        rd_objstr: object to read data from
        rd_is: indices to read from @rd_obj

        returns output as an numpy array
        """
        if self.xbar_m is None:
            raise RuntimeError("core xbar matrix is undefined")

        if rd_objstr not in self.objs:
            raise ValueError("object %s does not exist in this core" % (rd_objstr,))
        rd_obj = self.objs[rd_objstr]

        # Fill-in input vector for mxv
        x = np.zeros(shape=(len(rd_is),) )
        for i,idx in enumerate(rd_is):
            assert isinstance(idx, tuple) and len(idx) == len(rd_obj.shape), "idx=%s rd_obj.shape=%s" % (idx, rd_obj.shape)
            x[i] = rd_obj[idx]

        y = np.matmul(self.xbar_m, x)
        return y

    def write_obj(self, objname: str, w_idx, w_val):
        self.objs[objname][w_idx] = w_val

class Pipeline:
    def __init__(self,
                 stages: typing.List[Stage],
                 objs_shape: typing.Optional[typing.Dict[str, typing.Tuple[int, ...]]] = None,
                 execute_ops: bool = False):
        """ Initialize a pipeline

        stages: stages of the pipeline.
        objs_shape: the shapes of the objects accessed in stages.
        execute_ops: actually perform the operations on elements
        """

        if execute_ops and objs_shape is None:
            raise ValueError("execute_ops is set, but no objs_shape is given")

        for st in stages:
            # set the pipeline_write function for all stages
            st.pipeline_write = functools.partial(self.handle_write, st)
            # add cores to stages (for execution)
            st.core = Core()
            # set execute_ops
            st.execute_ops = execute_ops

        # Discover dependencies and build the loc_to_max_iter relation for
        # every writer/reader pair.
        readers = {} # object -> stage
        writers = {} # object -> stage
        for st in stages:
            rd_obj = st.get_rd_obj_name()
            if rd_obj is not None:
                # Sanity checks
                if rd_obj in readers:
                    raise ValueError("Object %s has more than one readers: %s,%s" % (rd_obj, st.get_name(), readers[rd_obj].get_name()))
                if objs_shape is not None and rd_obj not in objs_shape:
                    raise ValueError("Object %s not found in Pipeline initialization args" % (rd_obj,))
                readers[rd_obj] = st

            wr_obj = st.get_wr_obj_name()
            if wr_obj is not None:
                # NB: it might be possible to have more than one writers per
                # object, assuming they operate on different locations. For
                # simplicity, we assume one writer per object for now, and we
                # can fix this as needed.
                if wr_obj in writers:
                    raise ValueError("Object %s has more than one writers: %s,%s" % (wr_obj, st.get_name(), writers[rd_obj].get_name()))
                if objs_shape is not None and wr_obj not in objs_shape:
                    raise ValueError("Object %s not found in Pipeline initialization args" % (wr_obj,))
                writers[wr_obj] = st

        #assert(len(readers) == len(writers))
        for obj in readers.keys():
            reader = readers[obj]
            reader.set_rd_obj(obj)
            if obj not in writers:
                print("Object %s read by %s has no writters. Assuming it always exists." % (obj, reader.get_name()))
                reader.set_dont_wait_for_reads()
            else:
                writer = writers[obj]
                writer.set_wr_obj(obj)
                print("Object %s written by %s and read by %s" % (obj, writer.get_name(), reader.get_name()))
                loc_to_max_iter = isl_rel_loc_to_max_iter(writer.si.wr_a, reader.si.rd_a)
                reader.set_isl_rel_loc_to_max_iter(loc_to_max_iter)


        # Objects are stored in the core they are read.
        # Objects with no readers (orphans) are stored in this class.
        if objs_shape is not None:
            self.orphan_objs = {}
            for (obj,shape) in objs_shape.items():
                if obj in readers:
                    readers[obj].core.alloc_object(obj, shape)
                else:
                    # Assertion below cannot trigger because we are using a dict,
                    # but keep it in case code changes
                    assert obj not in self.orphan_objs
                    self.orphan_objs[obj] = np.zeros(shape)
                    if obj not in writers:
                        print("WARNING: object %s not read or written" % (obj,))
                    else:
                        print("Object %s is orphan: written by %s, but has no readers" % (obj, writers[obj].get_name()))


        # Now that we've set loc_to_max_iter, build the python module
        for st in stages:
            st.build_module()

        # Start the generators for every stage
        self.stage_ticks = [ (s,s.tick_gen()) for s in stages ]
        self.nticks = 0

        self.stages = stages
        self.readers = readers
        self.writers = writers
        # Writes buffer
        self.writes = []

    def configure(self, corecnfs: typing.List[CoreConf]):
        """ Configure the pipeline

        corecnfs: configuration of cores (expected in the same order as stages)
        """
        for (cconf,stage) in zip(corecnfs, self.stages):
            stage.core.configure(cconf)

    def handle_write(self, stage, wr_obj, wr_idx, wr_val):
        assert stage == self.writers[wr_obj]
        self.writes.append((wr_obj, wr_idx, wr_val))

    def flush_writes(self):
        for (wr_objstr, wr_idx, wr_val) in self.writes:
            reader = self.readers.get(wr_objstr, None)
            if reader is not None:
                # execute callback on the reader
                reader.write_callback(wr_objstr, wr_idx, wr_val)
            elif wr_val is not None and wr_objstr in self.orphan_objs:
                # if wr_val exsits, and this is an orphan object, update value
                wr_obj = self.orphan_objs[wr_objstr]
                assert isinstance(wr_idx, tuple) and len(wr_obj.shape) == len(wr_idx)
                wr_obj[wr_idx] = wr_val

        self.writes = []

    def get_object(self, objstr):
        if objstr in self.readers:
            return self.readers[objstr].core.get_object(objstr)
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
