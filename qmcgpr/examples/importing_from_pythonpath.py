#!/usr/bin/env python

from qmcgpr import testimport as test


def main():
    # To forgo using the try and except block, run this command
    # before using the code. Alternatively, the user rc file may be
    # updated to export the path automatically.
    # export PYTHONPATH=/PATH_TO_INSTALL/DMQMC-GPR/qmcgpr/:$PYTHONPATH
    test()
    return


if __name__ == '__main__':
    main()
