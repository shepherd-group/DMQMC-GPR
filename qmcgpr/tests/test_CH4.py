#!/usr/bin/env python

import os
import sys
import numpy
import pandas
import pkgutil
import unittest

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
        self.nloops = 5
        self.iloops = range(1, self.nloops + 1)
        self.beta_nslice = 10
        self.beta_cutoff = 0.1
        self.temp_nslice = 10
        self.temp_cutoff = 0.3
        self.fci = -0.18882 + -40.19868633
        self.dmqmc_data = DataDMQMC(
            [f'CH4/0{iloop}-DMQMC.out' for iloop in self.iloops]
        )
        dmqmc_beta, dmqmc_energy = self.dmqmc_data.get_data_for_training(
            key='E',
            nslice=self.beta_nslice,
            target_beta=0.0,
        )
        self.pip_data = DataDMQMC(
            [f'CH4/0{iloop}-PIPDMQMC.out' for iloop in self.iloops]
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
    def __init__(self) -> None:
        self.data = pandas.read_csv(
            'CH4/benchmark_gpr_data.csv'
        )
        self.betafit_full = numpy.load(
            'CH4/benchmark_betafit_full_domain.npy'
        )
        self.betafit_sub = numpy.load(
            'CH4/benchmark_betafit_sub_domain.npy'
        )
        self.tempfit_full = numpy.load(
            'CH4/benchmark_tempfit_full_domain.npy'
        )
        self.tempfit_sub = numpy.load(
            'CH4/benchmark_tempfit_sub_domain.npy'
        )


class TestRestartGeneration(unittest.TestCase, BenchmarkData, SystemData):
    ''' Create restart files given QMC data and check it matches
    benchmarks for the same data.
    '''
    def setUp(self):
        ''' Get the reference data
        '''
        SystemData.__init__(self)
        BenchmarkData.__init__(self)

    def doCleanups(self):
        ''' Clean up any files that could have been made while testing.'''
        if os.path.isfile('CH4/betafit_full_domain.npy'):
            os.remove('CH4/betafit_full_domain.npy')
        if os.path.isfile('CH4/tempfit_full_domain.npy'):
            os.remove('CH4/tempfit_full_domain.npy')
        if os.path.isfile('CH4/betafit_sub_domain.npy'):
            os.remove('CH4/betafit_sub_domain.npy')
        if os.path.isfile('CH4/tempfit_sub_domain.npy'):
            os.remove('CH4/tempfit_sub_domain.npy')

    def test_full_betafit(self):
        ''' Test the benchmark array matches the newly generated array.'''
        x = self.beta.copy()
        x = numpy.concatenate([x, [50.0 for _ in self.iloops]], axis=0)
        y = self.energy.copy()
        y = numpy.concatenate([y, [self.fci for _ in self.iloops]], axis=0)
        GPyDMQMC(
            x,
            y,
            self.beta,
            save_file='CH4/betafit_full_domain.npy',
            kernel_product=True,
        )
        betafit_full = numpy.load('CH4/betafit_full_domain.npy')
        self.assertTrue(numpy.array_equal(self.betafit_full, betafit_full))

    def test_full_tempfit(self):
        ''' Test the benchmark array matches the newly generated array.'''
        x = self.temp.copy()
        x = numpy.concatenate([x, [0.0 for _ in self.iloops]], axis=0)
        mask = numpy.isfinite(x)
        y = self.energy.copy()
        y = numpy.concatenate([y, [self.fci for _ in self.iloops]], axis=0)
        GPyDMQMC(
            x[mask],
            y[mask],
            self.temp[1:],
            save_file='CH4/tempfit_full_domain.npy',
            kernel_product=True,
        )
        tempfit_full = numpy.load('CH4/tempfit_full_domain.npy')
        self.assertTrue(numpy.array_equal(self.tempfit_full, tempfit_full))

    def test_sub_betafit(self):
        ''' Test the benchmark array matches the newly generated array.'''
        x = self.beta.copy()
        x = numpy.concatenate([x, [50.0 for _ in self.iloops]], axis=0)
        mask = numpy.divide(1.0, x) >= self.beta_cutoff
        y = self.energy.copy()
        y = numpy.concatenate([y, [self.fci for _ in self.iloops]], axis=0)
        GPyDMQMC(
            x[mask],
            y[mask],
            self.beta,
            save_file='CH4/betafit_sub_domain.npy',
            kernel_product=True,
        )
        betafit_sub = numpy.load('CH4/betafit_sub_domain.npy')
        self.assertTrue(numpy.array_equal(self.betafit_sub, betafit_sub))

    def test_sub_tempfit(self):
        ''' Test the benchmark array matches the newly generated array.'''
        x = self.temp.copy()
        x = numpy.concatenate([x, [0.0 for _ in self.iloops]], axis=0)
        mask = x < self.temp_cutoff
        y = self.energy.copy()
        y = numpy.concatenate([y, [self.fci for _ in self.iloops]], axis=0)
        GPyDMQMC(
            x[mask],
            y[mask],
            self.temp[1:],
            save_file='CH4/tempfit_sub_domain.npy',
            kernel_product=True,
        )
        tempfit_sub = numpy.load('CH4/tempfit_sub_domain.npy')
        self.assertTrue(numpy.array_equal(self.tempfit_sub, tempfit_sub))


class TestModelPredictions(unittest.TestCase, BenchmarkData, SystemData):
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

    def test_full_betafit(self):
        ''' Test the predicted energy and specific heat match the benchmark.'''
        x = self.beta.copy()
        x = numpy.concatenate([x, [50.0 for _ in self.iloops]], axis=0)
        y = self.energy.copy()
        y = numpy.concatenate([y, [self.fci for _ in self.iloops]], axis=0)
        gpr = GPyDMQMC(
            x,
            y,
            self.beta_predict,
            kernel_product=True,
        )
        self.assertTrue(
            numpy.array_equal(
                self.data['E(Beta-GPR00)'].round(12),
                gpr.y.round(12),
            )
        )
        self.assertTrue(
            numpy.array_equal(
                self.data['Cv(Beta-GPR00)'].round(12),
                ((-(gpr.x**2))*gpr.dy).round(12),
            )
        )

    def test_full_tempfit(self):
        ''' Test the predicted energy and specific heat match the benchmark.'''
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
        )
        self.assertTrue(
            numpy.array_equal(
                self.data['E(T-GPR00)'].round(12),
                gpr.y.round(12),
                equal_nan=True,
            )
        )
        self.assertTrue(
            numpy.array_equal(
                self.data['Cv(T-GPR00)'].round(12),
                gpr.dy.round(12),
                equal_nan=True,
            )
        )

    def test_sub_betafit(self):
        ''' Test the predicted energy and specific heat match the benchmark.'''
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
        )
        self.assertTrue(
            numpy.array_equal(
                self.data['E(Beta-GPR01)'].round(12),
                gpr.y.round(12),
            )
        )
        self.assertTrue(
            numpy.array_equal(
                self.data['Cv(Beta-GPR01)'].round(12),
                ((-(gpr.x**2))*gpr.dy).round(12),
            )
        )

    def test_sub_tempfit(self):
        ''' Test the predicted energy and specific heat match the benchmark.'''
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
        )
        self.assertTrue(
            numpy.array_equal(
                self.data['E(T-GPR01)'].round(12),
                gpr.y.round(12),
                equal_nan=True,
            )
        )
        self.assertTrue(
            numpy.array_equal(
                self.data['Cv(T-GPR01)'].round(12),
                gpr.dy.round(12),
                equal_nan=True,
            )
        )
