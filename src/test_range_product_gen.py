import types
import ast as pyast
from pprint import pprint

import islpy as isl
import astor as pyastor

from pipeline import isl_map_to_pyfn

## Original  code for reference
##
## # Generate a single function for all accesses in this operation
## l = [] # list of (ty,objname) tuples
## s = None # structure (idx, obj1, ..., objn)
## prod = None
## for acc in op.accesses:
##     l.append((acc.a_ty, acc.get_obj_name()))
##     if prod is None:
##         prod = acc.access
##     else:
##         prod = prod.range_product(acc.access)
##     if s is None:
##         s = (acc.get_idx_ndims(), acc.get_obj_ndims())
##     else:
##         assert s[0] == acc.get_idx_ndims()
##         s += (acc.get_obj_ndims(),)

## fn_name = "%s_op%02d_%s" % (self.get_name(), op_id, op.op_ty)
## py = isl_map_to_pyfn(prod, fn_name, s=s)
## # Add a decorator with information about the yield (RD/WR, object)
## # Not strictly needed, but seems useful for debugging, etc.
## dec_call = pyast.parse("annotate_op(%s)" % (str(tuple(l)),)).body[0].value
## py.decorator_list.append(dec_call)
## body.append(py)

r1 = isl.Map("{ S[i] -> x[j]: 0 <= i < 1 and i <= j < i + 2}")
w1 = isl.Map("{ S[i] -> y[j]: 0 <= i < 1 and i <= j < i + 2}")

assert r1.domain() == w1.domain()
prod = r1.range_product(w1)
print(prod)
body = []
py = isl_map_to_pyfn(prod, "fn")
body.append(py)
ast_mod = pyast.Module(body=body)
print(pyastor.to_source(ast_mod))

pyast.fix_missing_locations(ast_mod)
code = compile(ast_mod, "<generated>", "exec")
ret = types.ModuleType("S")
exec(code, ret.__dict__)

for (idx, x) in ret.fn():
    print(idx, x)
