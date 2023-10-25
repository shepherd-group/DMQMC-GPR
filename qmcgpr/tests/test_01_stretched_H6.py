#!/usr/bin/env python

import os
import sys
import numpy
import pandas
import pkgutil
import unittest

try:
    from qmcgpr import GPyDMQMC
except ModuleNotFoundError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if not pkgutil.find_loader('qmcgpr'):
        sys.path.append(os.path.join(_script_dir, '../'))
    from qmcgpr import GPyDMQMC


class SystemData:
    ''' Data from HANDE-QMC and the manuscript parameters for performing GPR.
    '''
    def __init__(self) -> None:
        self.dmqmc_data = pandas.read_csv(
            '01_stretched_H6/pyhande_dmqmc_fta.csv'
        )
        self.beta_nslice = 10
        self.beta_cutoff = 0.3
        self.temp_nslice = 10
        self.temp_cutoff = 0.5
        self.fci = -3.114120827642
        self.beta = self.dmqmc_data['Beta'].values.flatten()
        self.temp = numpy.divide(1.0, self.beta)
        self.energy = self.dmqmc_data['Tr[Hp]/Tr[p]'].values.flatten()


class BenchmarkData:
    ''' Pre-generated benchmark files.
    '''
    def __init__(self) -> None:
        self.data = pandas.read_csv(
            '01_stretched_H6/benchmark_gpr_data.csv'
        )
        self.betafit_full = numpy.load(
            '01_stretched_H6/benchmark_betafit_full_domain.npy'
        )
        self.betafit_sub = numpy.load(
            '01_stretched_H6/benchmark_betafit_sub_domain.npy'
        )
        self.tempfit_full = numpy.load(
            '01_stretched_H6/benchmark_tempfit_full_domain.npy'
        )
        self.tempfit_sub = numpy.load(
            '01_stretched_H6/benchmark_tempfit_sub_domain.npy'
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
        if os.path.isfile('01_stretched_H6/betafit_full_domain.npy'):
            os.remove('01_stretched_H6/betafit_full_domain.npy')
        if os.path.isfile('01_stretched_H6/tempfit_full_domain.npy'):
            os.remove('01_stretched_H6/tempfit_full_domain.npy')
        if os.path.isfile('01_stretched_H6/betafit_sub_domain.npy'):
            os.remove('01_stretched_H6/betafit_sub_domain.npy')
        if os.path.isfile('01_stretched_H6/tempfit_sub_domain.npy'):
            os.remove('01_stretched_H6/tempfit_sub_domain.npy')

    def test_full_betafit(self):
        ''' Test the benchmark array matches the newly generated array.'''
        x = self.beta[::self.beta_nslice]
        x = numpy.concatenate([x, [50.0]], axis=0)
        y = self.energy[::self.beta_nslice]
        y = numpy.concatenate([y, [self.fci]], axis=0)
        GPyDMQMC(
            x,
            y,
            self.beta,
            save_file='01_stretched_H6/betafit_full_domain.npy',
        )
        betafit_full = numpy.load('01_stretched_H6/betafit_full_domain.npy')
        self.assertTrue(numpy.array_equal(self.betafit_full, betafit_full))

    def test_full_tempfit(self):
        ''' Test the benchmark array matches the newly generated array.'''
        x = self.temp[::self.temp_nslice][1:]
        x = numpy.concatenate([x, [0.0]], axis=0)
        y = self.energy[::self.temp_nslice][1:]
        y = numpy.concatenate([y, [self.fci]], axis=0)
        GPyDMQMC(
            x,
            y,
            self.temp[1:],
            save_file='01_stretched_H6/tempfit_full_domain.npy',
        )
        tempfit_full = numpy.load('01_stretched_H6/tempfit_full_domain.npy')
        self.assertTrue(numpy.array_equal(self.tempfit_full, tempfit_full))

    def test_sub_betafit(self):
        ''' Test the benchmark array matches the newly generated array.'''
        x = self.beta[::self.beta_nslice]
        x = numpy.concatenate([x, [50.0]], axis=0)
        mask = numpy.divide(1.0, x) >= self.beta_cutoff
        y = self.energy[::self.beta_nslice]
        y = numpy.concatenate([y, [self.fci]], axis=0)
        GPyDMQMC(
            x[mask],
            y[mask],
            self.beta,
            save_file='01_stretched_H6/betafit_sub_domain.npy',
        )
        betafit_sub = numpy.load('01_stretched_H6/betafit_sub_domain.npy')
        self.assertTrue(numpy.array_equal(self.betafit_sub, betafit_sub))

    def test_sub_tempfit(self):
        ''' Test the benchmark array matches the newly generated array.'''
        x = self.temp[::self.temp_nslice][1:]
        x = numpy.concatenate([x, [0.0]], axis=0)
        mask = x < self.temp_cutoff
        y = self.energy[::self.temp_nslice][1:]
        y = numpy.concatenate([y, [self.fci]], axis=0)
        GPyDMQMC(
            x[mask],
            y[mask],
            self.temp[1:],
            save_file='01_stretched_H6/tempfit_sub_domain.npy',
        )
        tempfit_sub = numpy.load('01_stretched_H6/tempfit_sub_domain.npy')
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

        self.beta_predict = numpy.arange(0.0, 25.01, 0.01)
        self.temp_predict = self.beta_predict.copy()
        self.temp_predict[0] = 1E-8
        self.temp_predict = numpy.divide(1.0, self.temp_predict)

    def test_full_betafit(self):
        ''' Test the predicted energy and specific heat match the benchmark.'''
        x = self.beta[::self.beta_nslice]
        x = numpy.concatenate([x, [50.0]], axis=0)
        y = self.energy[::self.beta_nslice]
        y = numpy.concatenate([y, [self.fci]], axis=0)
        gpr = GPyDMQMC(
            x,
            y,
            self.beta_predict,
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
        x = self.temp[::self.temp_nslice][1:]
        x = numpy.concatenate([x, [0.0]], axis=0)
        y = self.energy[::self.temp_nslice][1:]
        y = numpy.concatenate([y, [self.fci]], axis=0)
        gpr = GPyDMQMC(
            x,
            y,
            self.temp_predict,
        )
        self.assertTrue(
            numpy.array_equal(
                self.data['E(T-GPR00)'].round(12),
                gpr.y.round(12),
            )
        )
        self.assertTrue(
            numpy.array_equal(
                self.data['Cv(T-GPR00)'].round(12),
                gpr.dy.round(12),
            )
        )

    def test_sub_betafit(self):
        ''' Test the predicted energy and specific heat match the benchmark.'''
        x = self.beta[::self.beta_nslice]
        x = numpy.concatenate([x, [50.0]], axis=0)
        mask = numpy.divide(1.0, x) >= self.beta_cutoff
        y = self.energy[::self.beta_nslice]
        y = numpy.concatenate([y, [self.fci]], axis=0)
        gpr = GPyDMQMC(
            x[mask],
            y[mask],
            self.beta_predict,
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
        x = self.temp[::self.temp_nslice][1:]
        x = numpy.concatenate([x, [0.0]], axis=0)
        mask = x < self.temp_cutoff
        y = self.energy[::self.temp_nslice][1:]
        y = numpy.concatenate([y, [self.fci]], axis=0)
        gpr = GPyDMQMC(
            x[mask],
            y[mask],
            self.temp_predict,
        )
        self.assertTrue(
            numpy.array_equal(
                self.data['E(T-GPR01)'].round(12),
                gpr.y.round(12),
            )
        )
        self.assertTrue(
            numpy.array_equal(
                self.data['Cv(T-GPR01)'].round(12),
                gpr.dy.round(12),
            )
        )
