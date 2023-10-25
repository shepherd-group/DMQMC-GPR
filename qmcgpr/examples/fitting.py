#!/usr/bin/env python

import os
import sys
import numpy
import pkgutil

try:
    from qmcgpr import GPyDMQMC
except ModuleNotFoundError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if not pkgutil.find_loader('qmcgpr'):
        sys.path.append(os.path.join(_script_dir, '../'))
    from qmcgpr import GPyDMQMC


def main():
    # Generate some example data for a simple eigenspectrum
    # Apply some noise to the energy, and fit with GPR.
    Ei = numpy.array([-1.3, -0.45, 0.3, 1.1])
    beta = numpy.arange(0.0, 15.0 + 0.05, 0.05)

    energy = numpy.zeros(beta.shape)

    for i, b in enumerate(beta):
        zi = numpy.exp(-b*Ei)
        Z = zi.sum()
        energy[i] = numpy.dot(zi/Z, Ei)

    numpy.random.seed(7)

    noisey_energy = numpy.random.normal(energy, scale=0.05)

    model = GPyDMQMC(
        beta,
        noisey_energy,
        beta[::10],
    )

    prediction_mask = numpy.isin(beta, model.x)

    masked_energy = energy[prediction_mask]
    masked_noisey_energy = noisey_energy[prediction_mask]

    print()
    print(
        f'{"Beta":>8} '
        f'{"ft-FCI":>16}'
        f'{"ft-FCI+Noise":>16}'
        f'{"GPR":>16}'
        f'{"GPR variance":>16}'
    )

    for i, b in enumerate(model.x):
        print(
            f'{b:>8.4f} '
            f'{masked_energy[i]:> 16.12f}'
            f'{masked_noisey_energy[i]:> 16.12f}'
            f'{model.y[i]:> 16.12f}'
            f'{model.var[i]:> 16.12f}'
        )

    return


if __name__ == '__main__':
    main()
