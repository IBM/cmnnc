# Copyright (c) 2019-2020, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

""" Misc utilities """

import typeguard
import typing


class xdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class xparams(xdict):
    def eval(self, expr):
        """ Evaluate an expression with the parameters """
        return eval(expr, None, self)

    def compute(self, p, expr):
        """ set a parameter based on an expression"""
        assert p not in self
        self[p] = self.eval(expr)


def check_class_hints(obj):
    """ Verify that the class type hints of an object are satisfied """
    hints = typing.get_type_hints(obj.__class__)
    for (name, hint_ty) in hints.items():
        val = getattr(obj, name)
        # # avoid bug in check_type()
        # # (it might have been fixed in later versions)
        # if hint_ty == Operation:
        #     if issubclass(val, hint_ty):
        #         continue
        typeguard.check_type(name, val, hint_ty)
