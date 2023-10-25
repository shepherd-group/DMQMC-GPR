#!/usr/bin/env python

import os
import sys
import pkgutil

try:
    from qmcgpr import DataDMQMC
except ModuleNotFoundError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if not pkgutil.find_loader('qmcgpr'):
        sys.path.append(os.path.join(_script_dir, '../'))
    from qmcgpr import DataDMQMC


def main():
    # Use individual beta loop files from the tests for Methane.
    outputs = [
        '../tests/CH4/01-DMQMC.out',
        '../tests/CH4/02-DMQMC.out',
        '../tests/CH4/03-DMQMC.out',
        '../tests/CH4/04-DMQMC.out',
        '../tests/CH4/05-DMQMC.out',
    ]

    data = DataDMQMC(outputs)

    # The x and y can then be provided to train the GPR model.
    x, y = data.get_data_for_training(
        key='E',
        nslice=10,
        target_beta=0.0,
    )

    print(
        f'{"Beta":>6} '
        f'{"E(Beta)":>16}'
    )

    for b, e in zip(x, y):
        print(
            f'{b:>6.3f} '
            f'{e:> 16.12f}'
        )

    return


if __name__ == '__main__':
    main()
