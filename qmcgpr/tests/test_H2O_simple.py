#!/usr/bin/env python

import os
import sys
import numpy
import pandas
import pkgutil
import unittest

from test_utils import TestDataTolerance

try:
    from qmcgpr import GPyDMQMC, DataDMQMC
except ModuleNotFoundError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if not pkgutil.find_loader('qmcgpr'):
        sys.path.append(os.path.join(_script_dir, '../'))
    from qmcgpr import GPyDMQMC, DataDMQMC


class SystemData:
    ''' Data from HANDE-QMC and the manuscript parameters for performing GPR.
    '''
    def __init__(self) -> None:
        self.nloops = 4
        self.iloops = range(1, self.nloops + 1)
        self.xo_temp = 1.3
        self.beta_nslice = 10
        self.beta_cutoff = 0.6
        self.temp_nslice = 10
        self.temp_cutoff = 4.0
        self.fci = -0.216736 + -76.02588824
        self.dmqmc_data = DataDMQMC(
            [f'H2O/0{iloop}-DMQMC.out' for iloop in self.iloops]
        )
        dmqmc_beta, dmqmc_energy = self.dmqmc_data.get_data_for_training(
            key='E',
            nslice=self.beta_nslice,
            target_beta=0.0,
        )
        self.pip_data = DataDMQMC(
            [f'H2O/0{iloop}-PIPDMQMC.out' for iloop in self.iloops]
        )
        pip_beta, pip_energy = self.pip_data.get_data_for_training(
            key='E',
            nslice=self.temp_nslice,
            target_beta=1.0,
        )
        self.beta = numpy.concatenate(
            [dmqmc_beta, pip_beta],
            axis=0,
        )
        self.temp = numpy.divide(1.0, self.beta)
        self.energy = numpy.concatenate(
            [dmqmc_energy, pip_energy],
            axis=0,
        )


class BenchmarkData:
    ''' Pre-generated benchmark files.
    '''
    compare_data = TestDataTolerance()

    def __init__(self) -> None:
        self.data = pandas.read_csv(
            'H2O/benchmark_gpr_data.csv'
        )


class TestRestartPredictions(unittest.TestCase, BenchmarkData, SystemData):
    ''' Create restart files given QMC data and check it matches
    benchmarks for the same data.
    '''
    def setUp(self):
        ''' Get the reference data
        '''
        SystemData.__init__(self)
        BenchmarkData.__init__(self)

        self.beta_predict = numpy.arange(0.0, 30.01, 0.01)
        self.temp_predict = self.beta_predict.copy()
        self.temp_predict[0] = numpy.nan
        self.temp_predict = numpy.divide(1.0, self.temp_predict)
        self.beta_xo_mask = self.temp_predict >= self.xo_temp
        self.temp_xo_mask = self.temp_predict < self.xo_temp

    def test_full_betafit(self):
        ''' Test the benchmark array matches the newly generated array.'''
        x = self.beta.copy()
        x = numpy.concatenate([x, [50.0 for _ in self.iloops]], axis=0)
        y = self.energy.copy()
        y = numpy.concatenate([y, [self.fci for _ in self.iloops]], axis=0)
        gpr = GPyDMQMC(
            x,
            y,
            self.beta_predict,
            kernel_product=True,
            restart_file='H2O/benchmark_betafit_full_domain.npy',
        )
        self.assertTrue(self.compare_data[
            self.data['E(Beta-GPR00)'].values,
            gpr.y,
            self.beta_xo_mask,
        ])
        self.assertTrue(self.compare_data[
            self.data['Cv(Beta-GPR00)'].values,
            ((-(gpr.x**2))*gpr.dy),
            self.beta_xo_mask,
        ])

    def test_full_tempfit(self):
        ''' Test the benchmark array matches the newly generated array.'''
        x = self.temp.copy()
        x = numpy.concatenate([x, [0.0 for _ in self.iloops]], axis=0)
        mask = numpy.isfinite(x)
        y = self.energy.copy()
        y = numpy.concatenate([y, [self.fci for _ in self.iloops]], axis=0)
        gpr = GPyDMQMC(
            x[mask],
            y[mask],
            self.temp_predict,
            kernel_product=True,
            restart_file='H2O/benchmark_tempfit_full_domain.npy',
        )
        self.assertTrue(self.compare_data[
            self.data['E(T-GPR00)'].values,
            gpr.y,
            self.temp_xo_mask,
        ])
        self.assertTrue(self.compare_data[
            self.data['Cv(T-GPR00)'].values,
            gpr.dy,
            self.temp_xo_mask,
        ])

    def test_sub_betafit(self):
        ''' Test the benchmark array matches the newly generated array.'''
        x = self.beta.copy()
        x = numpy.concatenate([x, [50.0 for _ in self.iloops]], axis=0)
        mask = numpy.divide(1.0, x) >= self.beta_cutoff
        y = self.energy.copy()
        y = numpy.concatenate([y, [self.fci for _ in self.iloops]], axis=0)
        gpr = GPyDMQMC(
            x[mask],
            y[mask],
            self.beta_predict,
            kernel_product=True,
            restart_file='H2O/benchmark_betafit_sub_domain.npy',
        )
        self.assertTrue(self.compare_data[
            self.data['E(Beta-GPR01)'].values,
            gpr.y,
            self.beta_xo_mask,
        ])
        self.assertTrue(self.compare_data[
            self.data['Cv(Beta-GPR01)'].values,
            ((-(gpr.x**2))*gpr.dy),
            self.beta_xo_mask,
        ])

    def test_sub_tempfit(self):
        ''' Test the benchmark array matches the newly generated array.'''
        x = self.temp.copy()
        x = numpy.concatenate([x, [0.0 for _ in self.iloops]], axis=0)
        mask = numpy.isfinite(x)
        mask = numpy.logical_and(mask, x < self.temp_cutoff)
        y = self.energy.copy()
        y = numpy.concatenate([y, [self.fci for _ in self.iloops]], axis=0)
        gpr = GPyDMQMC(
            x[mask],
            y[mask],
            self.temp_predict,
            kernel_product=True,
            restart_file='H2O/benchmark_tempfit_sub_domain.npy',
        )
        self.assertTrue(self.compare_data[
            self.data['E(T-GPR01)'].values,
            gpr.y,
            self.temp_xo_mask,
        ])
        self.assertTrue(self.compare_data[
            self.data['Cv(T-GPR01)'].values,
            gpr.dy,
            self.temp_xo_mask,
        ])
