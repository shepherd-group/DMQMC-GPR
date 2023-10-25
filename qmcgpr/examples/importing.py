#!/usr/bin/env python

import os
import sys
import pkgutil

try:
    from qmcgpr import testimport as test
except ModuleNotFoundError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if not pkgutil.find_loader('qmcgpr'):
        sys.path.append(os.path.join(_script_dir, '../'))
    from qmcgpr import testimport as test


def main():
    test()
    return


if __name__ == '__main__':
    main()
