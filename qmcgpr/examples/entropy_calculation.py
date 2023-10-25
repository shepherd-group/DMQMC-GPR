#!/usr/bin/env python

import os
import sys
import pandas
import pkgutil

try:
    from qmcgpr import form_combined_dataset, numerical_entropy
except ModuleNotFoundError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if not pkgutil.find_loader('qmcgpr'):
        sys.path.append(os.path.join(_script_dir, '../'))
    from qmcgpr import form_combined_dataset, numerical_entropy


def main():
    # Get the specific heat data from the tests for water.
    data = pandas.read_csv('../tests/H2O/benchmark_gpr_data.csv')

    # Using the crossover temperature, form the combined data set.
    crossover = 1.3

    # Get the data we need to make the combined data set.
    temperature = data['T'].values.flatten()
    temperature_CV = data['Cv(T-GPR01)'].values.flatten()
    beta = data['Beta'].values.flatten()
    beta_CV = data['Cv(Beta-GPR01)'].values.flatten()

    # Form the combined data set.
    temperature, specific_heat = form_combined_dataset(
        crossover,
        temperature,
        temperature_CV,
        beta,
        beta_CV,
    )

    # Generate out entropy estimate.
    T, S = numerical_entropy(
        temperature,
        specific_heat,
    )

    print(
        f'{"T":>8} '
        f'{"S(T)":>16}'
    )

    for t, s in zip(T[::100], S[::100]):
        print(
            f'{t:>8.4f} '
            f'{s:> 16.12f}'
        )

    return


if __name__ == '__main__':
    main()
