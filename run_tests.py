#!/usr/bin/env python

# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import sys
import pytest

sys.path.append("./src")

if __name__ == '__main__':
    args  = []
    args.extend(["-W","ignore::DeprecationWarning"])
    #args.extend(["-s"])
    #args.extend(["--full-trace"])
    args.extend(sys.argv[1:])
    sys.exit(pytest.main(args))
