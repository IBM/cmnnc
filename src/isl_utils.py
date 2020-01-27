# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:


import typing
import ast as pyast

import islpy as isl

# import ast as pyast
# import astor as pyastor
# from astpp import parseprint, dump as astpp_dump

# TODO: add an isl prefix to all functions (that do not have one)

## ISL Helpers


def isl_fix_params(x, vals):
    """ Fix the value of the parameters in @x according to the @vals dict.

    Return result
    """
    ret = x
    for (vp, vv) in vals.items():
        pos = x.find_dim_by_name(isl.dim_type.param, vp)
        if pos == -1:
            # ValueError("param %s does not exist" % (vp,))
            continue
        ret = ret.fix_val(isl.dim_type.param, pos, vv)
        # ret = ret.add_constraint(isl.Constraint.eq_from_names(ret.space, {vp: 1, 1: vv}))
    return ret


def str_to_isl_map(x: str) -> isl.Map:
    try:
        return isl.Map(x)
    except:
        print("Failed to create an isl.Map from %s" % (x,))
        raise


def dict_from_map(isl_map, p_key, p_val):
    """ Create a dictionary from an ISL map. """
    ret = {}

    def add_dep(p):
        p_var_dict = p.get_var_dict()
        if p_key not in p_var_dict:
            raise ValueError(
                "p_key='%s' not in %s" % (p_key, ",".join(p_var_dict.keys()))
            )
        if p_val not in p_var_dict:
            raise ValueError(
                "p_val='%s' not in %s" % (p_key, ",".join(p_var_dict.keys()))
            )

        k = p.get_coordinate_val(*p_var_dict[p_key]).to_python()
        v = p.get_coordinate_val(*p_var_dict[p_val]).to_python()
        if k not in ret:
            ret[k] = []
        ret[k].append(v)

    isl_map.wrap().foreach_point(add_dep)
    return ret


def isl_rel_loc_to_max_iter(s1_wr_a, s2_rd_a):
    """
    Let's assume that we have two loops where one writes to an object and
    another reads from that object. Let's also assume that the first loop writes
    locations to the object only once. The instance space of the first loop is
    S1, the second S2, and the index space of the object is O.

    We want to compile a state machine that observes writes from the first
    loop, and advances the second loop respecting RAW dependencies. That is, it
    does not execute an iteration of the second loop until all the data that
    the iteration reads have been written by the first loop.

    To do that we create an ISL relation from O (i.e., object locations) to S2
    (i.e., the iteration space of the second loop) that maps observed writes to
    O to the maximum iteration in S2 that can be executed given that the observed
    write was performed.
    """
    #print("WR:%s\n%RD:%s" % (s1_wr_a, s2_rd_a))

    # First, we compute a relation S2 -> S1, such that o2 (in S2) -> o1 (in
    # S1), iff o2 reads something that is being written by o1.
    ret = s2_rd_a.apply_range(s1_wr_a.reverse())

    # We keep the maximum element, ie, the (lexigraphically) last instance
    # (iteration point) of o2 that need to be executed before each o1.
    # (This is probably not needed given what we do next)
    ret = ret.lexmax()

    # Now in theory we could have:
    # 0 -> 2
    # 1 -> 1
    # 2 -> 2
    #
    # I.e., iteration 0 depends on 2, and iteration 1 depends 1.
    # We want to transform this to:
    # 0 -> 2
    # 1 -> 2
    # 2 -> 2
    #
    # That is if we have D: S2 -> S1 such that o2 -> o1 iff o1 is the last
    # iteration that o2 has to wait for, then we want a new relation D':S2 ->
    # S1 such that o2 -> o1 iff o1 is the last iteration that j<=o2 has to wait
    # for.
    #
    # IOW, D'(i) = max(D(j): j <= i)
    #
    # We can compute this in ISL as follows:
    # D1 := dom D
    # D' := lexmax ((D1 >>= D1) before D)
    #
    # - D1 is the domain of D, i.e., S2
    # - D1 >>= D1 is a relation from D1 to all elements for which the source element is
    #   lexicographically greater than the destination elements.
    # - ((D1 >>= D1) before D)
    #
    # (For more details see prefix_max.py)
    d_ = ret.domain()
    ret = d_.lex_ge_set(d_).apply_range(ret).lexmax()

    # Next, we compute (D' before s1_wr_a), and subsequently reverse it to
    # produce a mapping of locations written by the first loop, to the maximum
    # iteration we can execute on the first loop
    ret = ret.apply_range(s1_wr_a).reverse()

    # Because we are reversing the relationship, a write might be mapped to
    # multiple iterations. Hence, the final lexmax
    ret = ret.lexmax()

    return ret


### Code generation


def print_ast(ast):
    global ast_node
    p = isl.Printer.to_str(isl.DEFAULT_CONTEXT)
    p = p.set_output_format(isl.format.C)
    p.flush()
    p = p.print_ast_node(ast)
    print(p.get_str())


def isl_set_to_ast(isl_s):
    """ ISL set to an AST """
    # According to the ISL manual:
    #
    #   In case the schedule given as a isl_union_map, an AST is generated that
    #   visits all the elements in the domain of the isl_union_map according to the
    #   lexicographic order of the corresponding image element(s). If the range of
    #   the isl_union_map consists of elements in more than one space, then each of
    #   these spaces is handled separately in an arbitrary order. It should be
    #   noted that the schedule tree or the image elements in a schedule map only
    #   specify the order in which the corresponding domain elements should be
    #   visited. No direct relation between the partial schedule values or the
    #   image elements on the one hand and the loop iterators in the generated AST
    #   on the other hand should be assumed.
    #
    # My guess is that the sched below is just the identity.
    sched = isl.Map.from_domain(isl_s)
    bld = isl.AstBuild.from_context(isl.Set("{:}"))
    try:
        ast = bld.ast_from_schedule(sched)
    except:
        print("Failed to build ISL AST for ISL map %s" % (isl_s,))
        raise
    return ast


def isl_map_to_ast(isl_m):
    """ ISL map to an AST """
    return isl_set_to_ast(isl_m.wrap())


## Python Code generation


def isl2py_exp(e):
    ty = e.get_type()  # returns isl_ast_expr_type
    # Expression
    if ty == isl.ast_expr_type.op:
        op_ty = e.get_op_type()
        # AND
        if op_ty == isl.ast_expr_op_type.and_:
            op0 = isl2py_exp(e.get_op_arg(0))
            op1 = isl2py_exp(e.get_op_arg(1))
            expr = pyast.BoolOp(pyast.And(), [op0, op1])
            return expr

        # LESS EQUAL
        elif op_ty == isl.ast_expr_op_type.le:
            op0 = isl2py_exp(e.get_op_arg(0))
            op1 = isl2py_exp(e.get_op_arg(1))
            expr = pyast.Compare(
                left=op0, ops=[pyast.LtE(),], comparators=[op1]
            )
            return expr

        # LESS
        elif op_ty == isl.ast_expr_op_type.lt:
            op0 = isl2py_exp(e.get_op_arg(0))
            op1 = isl2py_exp(e.get_op_arg(1))
            expr = pyast.Compare(left=op0, ops=[pyast.Lt(),], comparators=[op1])
            return expr

        # EQUALITY
        elif op_ty == isl.ast_expr_op_type.eq:
            op0 = isl2py_exp(e.get_op_arg(0))
            op1 = isl2py_exp(e.get_op_arg(1))
            expr = pyast.Compare(left=op0, ops=[pyast.Eq(),], comparators=[op1])
            return expr

        # GREATER EQUAL
        elif op_ty == isl.ast_expr_op_type.ge:
            op0 = isl2py_exp(e.get_op_arg(0))
            op1 = isl2py_exp(e.get_op_arg(1))
            expr = pyast.Compare(
                left=op0, ops=[pyast.GtE(),], comparators=[op1]
            )
            return expr

        # Minus
        elif op_ty == isl.ast_expr_op_type.minus:
            op0 = isl2py_exp(e.get_op_arg(0))
            expr = pyast.UnaryOp(pyast.USub(), op0)
            return expr

        # ADD
        elif op_ty == isl.ast_expr_op_type.add:
            op0 = isl2py_exp(e.get_op_arg(0))
            op1 = isl2py_exp(e.get_op_arg(1))
            expr = pyast.BinOp(op0, pyast.Add(), op1)
            return expr

        # SUB
        elif op_ty == isl.ast_expr_op_type.sub:
            op0 = isl2py_exp(e.get_op_arg(0))
            op1 = isl2py_exp(e.get_op_arg(1))
            expr = pyast.BinOp(op0, pyast.Sub(), op1)
            return expr

        # MUL
        elif op_ty == isl.ast_expr_op_type.mul:
            op0 = isl2py_exp(e.get_op_arg(0))
            op1 = isl2py_exp(e.get_op_arg(1))
            expr = pyast.BinOp(op0, pyast.Mult(), op1)
            return expr

        # MAX
        elif op_ty == isl.ast_expr_op_type.max:
            # NB: Not sure if max can actually have more than one args.
            nargs = e.get_op_n_arg()
            args = [isl2py_exp(e.get_op_arg(i)) for i in range(nargs)]
            expr = pyast.Call(
                func=pyast.Name(id="max", ctx=pyast.Load()),
                args=args,
                keywords=[],
            )
            return expr

        # MIN
        elif op_ty == isl.ast_expr_op_type.min:
            # NB: Not sure if max can actually have more than one args.
            nargs = e.get_op_n_arg()
            args = [isl2py_exp(e.get_op_arg(i)) for i in range(nargs)]
            expr = pyast.Call(
                func=pyast.Name(id="min", ctx=pyast.Load()),
                args=args,
                keywords=[],
            )
            return expr

        else:
            raise NotImplementedError("No support for op_ty=%d" % (op_ty,))
    # ID
    elif ty == isl.ast_expr_type.id:
        name = e.get_id().name
        if name == "":
            return pyast.NameConstant(None)
        else:
            return pyast.Name(name, pyast.Load())
    # INT
    elif ty == isl.ast_expr_type.int:
        val = e.get_val().to_python()
        return pyast.Num(val)
    elif ty == isl.ast_expr_type.error:
        raise NotImplementedError
    else:
        raise AssertionError("uknown ISL expr type: %d" % (ty,))


def isl2py_for(n):
    assert n.get_type() == isl.ast_node_type.for_
    for_var = n.for_get_iterator()
    assert for_var.get_type() == isl.ast_expr_type.id
    for_var_name = for_var.get_id().name

    # Initialize loop variable
    py_asign = pyast.Assign(
        targets=[pyast.Name(for_var_name, pyast.Store())],
        value=isl2py_exp(n.for_get_init()),
    )

    # Increment statement
    py_inc = pyast.AugAssign(
        target=pyast.Name(for_var_name, pyast.Store()),
        op=pyast.Add(),
        value=isl2py_exp(n.for_get_inc()),
    )

    # python loop body
    py_body = isl2py_ast(n.for_get_body()) + [py_inc]

    ret = [
        py_asign,
        pyast.While(test=isl2py_exp(n.for_get_cond()), body=py_body, orelse=[]),
    ]

    return ret


## This should return a list, i.e., a "body"
def isl2py_ast(n):
    """ Transform a ISL AST node to a list of Python AST nodes (body) """
    isl_nty = n.get_type()
    if isl_nty == isl.ast_node_type.for_:
        return isl2py_for(n)

    elif isl_nty == isl.ast_node_type.if_:
        return [
            pyast.If(
                test=isl2py_exp(n.if_get_cond()),
                body=isl2py_ast(n.if_get_then()),
                orelse=isl2py_ast(n.if_get_else()) if n.if_has_else() else [],
            )
        ]

    elif isl_nty == isl.ast_node_type.block:
        nlist = n.block_get_children()
        nr_nodes = nlist.n_ast_node()
        ret = []
        for i in range(nr_nodes):
            ret.extend(isl2py_ast(nlist.get_ast_node(i)))
        return ret
        raise NotImplementedError
    elif isl_nty == isl.ast_node_type.mark:
        raise NotImplementedError
    elif isl_nty == isl.ast_node_type.user:
        e = n.user_get_expr()
        if (
            e.get_type() == isl.ast_expr_type.op
            and e.get_op_type() == isl.ast_expr_op_type.call
        ):
            # Replace call with a yield with the call arguments
            # I guess the first arg is the "function"
            nargs = e.get_op_n_arg()
            return [
                # NB: Ignore the first argument (for now)
                pyast.Expr(
                    pyast.Yield(
                        pyast.Tuple(
                            [
                                isl2py_exp(e.get_op_arg(i + 1))
                                for i in range(nargs - 1)
                            ],
                            pyast.Load(),
                        )
                    )
                )
            ]
        raise NotImplementedError
    elif isl_nty == isl.ast_node_type.error:
        raise NotImplementedError
    else:
        raise AssertionError("uknown ISL node type: %d" % (isl_nty,))


def isl2py_fn(node, fn_name):
    """ Transform an ISL AST node to a Python AST function"""
    return pyast.FunctionDef(
        name=fn_name,
        args=pyast.arguments(
            args=[],
            defaults=[],
            vararg=None,
            kwonlyargs=[],
            kwarg=None,
            kw_defaults=[],
        ),
        body=isl2py_ast(node),
        decorator_list=[],
    )

def isl_set_from_names(
    tname: str,
    names: typing.List[str],
    ctx = isl.DEFAULT_CONTEXT
) -> isl.Set:
    space = isl.Space.create_from_names(
        ctx, set=names,
    ).set_tuple_name(isl.dim_type.set, tname)
    ret = isl.Set.universe(space)
    return ret

def isl_set_from_shape(
    tname: str,
    names: typing.List[str],
    shape: typing.Tuple[int,...],
    ctx = isl.DEFAULT_CONTEXT
) -> isl.Set:
    assert len(shape) == len(names)
    ret = isl_set_from_names(tname, names)
    ineq_from_names = isl.Constraint.ineq_from_names
    for (name, dim) in zip(names, shape):
        for c in (
            {1: 0, name: 1},
            {1: dim-1, name: -1},
        ):
            ret = ret.add_constraint(ineq_from_names(ret.space, c))
    return ret
