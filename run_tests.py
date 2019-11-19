#!/usr/bin/env python

import sys
import pytest

sys.path.append("./src")

if __name__ == '__main__':
    # sys.exit(pytest.main(["-W","ignore::DeprecationWarning","--full-trace", "-s"]))
    # sys.exit(pytest.main(["-W","ignore::DeprecationWarning", "-s"]))
    args  = []
    args.extend(["-W","ignore::DeprecationWarning"])
    #args.extend(["-s"])
    args.extend(sys.argv[1:])
    sys.exit(pytest.main(args))
