# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import astor as pyastor

import islpy as isl
from isl_utils import isl_map_to_ast, isl2py_fn
from pipeline import StructureTupleYields

def test_structure_yields():
    rel = isl.Map("{ A[i,j] -> B[k] : 0 <= i < 10 and 0 <= j < 3 and k = i + j }")
    rel_ast = isl_map_to_ast(rel)
    structure = (ndims_in, ndims_out) =  tuple(rel.space.dim(x) for x in (isl.dim_type.in_, isl.dim_type.out))
    py = isl2py_fn(rel_ast, "foo")
    print("Before:\n", pyastor.to_source(py))
    StructureTupleYields(structure).visit(py)
    print("After:\n", pyastor.to_source(py))
    # TODO: actually check the generated i
