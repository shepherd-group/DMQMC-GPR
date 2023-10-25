#!/usr/bin/env python

import os
import sys
import numpy
import pandas
import pkgutil

try:
    from qmcgpr import GPyDMQMC, form_combined_dataset, numerical_entropy
except ModuleNotFoundError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if not pkgutil.find_loader('qmcgpr'):
        sys.path.append(os.path.join(_script_dir, '../'))
    from qmcgpr import GPyDMQMC, form_combined_dataset, numerical_entropy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

mpl.rc('text', usetex=True)
mpl.rc('savefig', dpi=100)
mpl.rc('lines', lw=2, markersize=6)
mpl.rc('legend', fontsize=8, numpoints=1)
mpl.rc(('axes', 'xtick', 'ytick'), labelsize=8)
mpl.rc('figure', dpi=200, figsize=(3.37, 3.37*(numpy.sqrt(5)-1)/2))
mpl.rc('font', **{'family': 'serif', 'sans-serif': 'Computer Modern Roman'})


def main():
    # This example demonstrates generally the complete workflow for
    # estimating the specific heat and entropy using DMQMC data.
    # To save time the restart files are used for each model rather than
    # retraining.

    # Form our training and prediction data arrays.
    beta_predict = numpy.arange(0.0, 25.01, 0.01)

    temperature_predict = numpy.divide(1.0, beta_predict)
    temperature_predict[0] = 1E8

    dmqmc_data = pandas.read_csv('../tests/BeH2/pyhande_dmqmc_fta.csv')

    beta_train = numpy.concatenate(
        [[50.0], beta_predict[::10]],
        axis=0,
    )
    temperature_train = numpy.concatenate(
        [[0.0], temperature_predict[::10]],
        axis=0,
    )

    energy_train = dmqmc_data['Tr[Hp]/Tr[p]'].values.flatten()
    energy_train = numpy.concatenate(
        [[-15.816038172880], energy_train[::10]],
        axis=0,
    )

    # Load in the benchmark models from the test data and predict on those.
    beta_model = GPyDMQMC(
        beta_train,
        energy_train,
        beta_predict,
        restart_file='../tests/BeH2/benchmark_betafit_full_domain.npy',
    )

    beta_model.dy *= -(beta_model.x**2.0)

    temperature_model = GPyDMQMC(
        temperature_train,
        energy_train,
        temperature_predict,
        restart_file='../tests/BeH2/benchmark_tempfit_full_domain.npy',
    )

    retrained_beta_model = GPyDMQMC(
        beta_train[temperature_train >= 0.5],
        energy_train[temperature_train >= 0.5],
        beta_predict,
        restart_file='../tests/BeH2/benchmark_betafit_sub_domain.npy',
    )

    retrained_beta_model.dy *= -(retrained_beta_model.x**2.0)

    retrained_temperature_model = GPyDMQMC(
        temperature_train[temperature_train < 1.0],
        energy_train[temperature_train < 1.0],
        temperature_predict,
        restart_file='../tests/BeH2/benchmark_tempfit_sub_domain.npy',
    )

    # Using the crossover temperature, form the combined data set.
    crossover = 0.75

    temperature, specific_heat = form_combined_dataset(
        crossover,
        retrained_temperature_model.x,
        retrained_temperature_model.dy,
        retrained_beta_model.x,
        retrained_beta_model.dy,
    )

    # Generate out entropy estimate.
    T, S = numerical_entropy(
        temperature,
        specific_heat,
    )

    # Plot everything that was done.
    with PdfPages('Work-Flow-Example.pdf') as pdf:

        plt.clf()

        fig, (ax0, ax1, ax2) = plt.subplots(
            nrows=3,
            ncols=1,
            sharex=True,
            sharey=False,
        )

        # First time training
        ax0.plot(
                1.0/beta_model.x,
                beta_model.dy,
                color='C0',
                ls='-',
                zorder=3,
            )

        ax0.plot(
                temperature_model.x,
                temperature_model.dy,
                color='C1',
                ls='--',
                zorder=5,
            )

        # After training again on a user selected subsets
        ax1.plot(
                1.0/retrained_beta_model.x,
                retrained_beta_model.dy,
                color='C0',
                ls='-',
                zorder=3,
            )

        ax1.plot(
                retrained_temperature_model.x,
                retrained_temperature_model.dy,
                color='C1',
                ls='--',
                zorder=5,
            )

        # Final subsets combined
        ax2.plot(
                temperature[temperature >= crossover],
                specific_heat[temperature >= crossover],
                color='C0',
                ls='-',
                zorder=3,
                label=r'$\beta$ model',
            )

        ax2.plot(
                temperature[temperature < crossover],
                specific_heat[temperature < crossover],
                color='C1',
                ls='--',
                zorder=5,
                label=r'$T$ model',
            )

        ylim = ax0.get_ylim()
        ax1.set_ylim(ylim)

        ax0.set_ylabel(r'$C_{V}(T)$ / $k_{\mathrm{B}}$')
        ax1.set_ylabel(r'$C_{V}(T)$ / $k_{\mathrm{B}}$')
        ax2.set_ylabel(r'$C_{V}(T)$ / $k_{\mathrm{B}}$')

        ax2.set_xscale('log')
        ax2.set_xlim(1/25.0, 1/0.01)
        ax2.set_xlabel(r'$T$ / $E_{\mathrm{h}} \! \ k_{\mathrm{B}}^{-1}$')

        fig.subplots_adjust(hspace=0.0)
        plt.gcf().set_size_inches(3.37, 3*3.37*(numpy.sqrt(5)-1)/2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
        pdf.savefig(bbox_inches='tight')


        # Plot the entropy estimate from the combined estimate.
        plt.clf()

        plt.gcf().set_size_inches(3.37, 3.37*(numpy.sqrt(5)-1)/2)

        plt.plot(
                T,
                S,
                label=r'Numerical entropy',
                color='C2',
                ls='-',
            )

        plt.xlabel(r'$T$ / $E_{\mathrm{h}} \! \ k_{\mathrm{B}}^{-1}$')
        plt.xscale('log')
        plt.xlim(1/25.0, 1/0.01)

        plt.ylabel(r'$S(T)$ / $k_{\mathrm{B}}$')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
        pdf.savefig(bbox_inches='tight')

    return


if __name__ == '__main__':
    main()
