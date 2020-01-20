#!/usr/bin/env python

# Copyright (c) 2019, IBM Research.
#
# Author: Kornilios Kourtis <kou@zurich.ibm.com>
#
# vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

import sys
import pytest

sys.path.append("./src")

# If you want no messages:
#  ./run_tests.py --capture=fd --show-capture=no
# If you want to stop at the first failure:
#  ./run_tests.py --maxfail 1

if __name__ == '__main__':
    args  = []
    args.extend(["-W","ignore::DeprecationWarning"])
    #args.extend(["-s"])
    #args.extend(["--full-trace"])
    args.extend(sys.argv[1:])
    sys.exit(pytest.main(args))
