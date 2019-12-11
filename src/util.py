# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

""" Misc utilities """

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
