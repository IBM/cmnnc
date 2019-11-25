# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import itertools
import copy
import ast as pyast

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
