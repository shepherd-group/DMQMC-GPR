#!/usr/bin/env python

import os
import sys
import numpy
import pandas
import pkgutil
import unittest

from test_utils import TestDataTolerance

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
            'Be/pyhande_dmqmc_fta.csv'
        )
        self.xo_temp = 0.6
        self.beta_nslice = 10
        self.beta_cutoff = 0.5
        self.temp_nslice = 10
        self.temp_cutoff = 0.75
        self.fci = -14.617494507748
        self.beta = self.dmqmc_data['Beta'].values.flatten()
        self.temp = numpy.divide(1.0, self.beta)
        self.energy = self.dmqmc_data['Tr[Hp]/Tr[p]'].values.flatten()


class BenchmarkData:
    ''' Pre-generated benchmark files.
    '''
    compare_data = TestDataTolerance()

    def __init__(self) -> None:
        self.data = pandas.read_csv(
            'Be/benchmark_gpr_data.csv'
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

        self.beta_predict = numpy.arange(0.0, 25.01, 0.01)
        self.temp_predict = self.beta_predict.copy()
        self.temp_predict[0] = 1E-8
        self.temp_predict = numpy.divide(1.0, self.temp_predict)
        self.beta_xo_mask = self.temp_predict >= self.xo_temp
        self.temp_xo_mask = self.temp_predict < self.xo_temp

    def test_full_betafit(self):
        ''' Test the benchmark array matches the newly generated array.'''
        x = self.beta[::self.beta_nslice]
        x = numpy.concatenate([[50.0], x], axis=0)
        y = self.energy[::self.beta_nslice]
        y = numpy.concatenate([[self.fci], y], axis=0)
        gpr = GPyDMQMC(
            x,
            y,
            self.beta_predict,
            restart_file='Be/benchmark_betafit_full_domain.npy',
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
        x = self.temp[::self.temp_nslice][1:]
        x = numpy.concatenate([[0.0], x], axis=0)
        y = self.energy[::self.temp_nslice][1:]
        y = numpy.concatenate([[self.fci], y], axis=0)
        gpr = GPyDMQMC(
            x,
            y,
            self.temp_predict,
            restart_file='Be/benchmark_tempfit_full_domain.npy',
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
        x = self.beta[::self.beta_nslice]
        x = numpy.concatenate([[50.0], x], axis=0)
        mask = numpy.divide(1.0, x) >= self.beta_cutoff
        y = self.energy[::self.beta_nslice]
        y = numpy.concatenate([[self.fci], y], axis=0)
        gpr = GPyDMQMC(
            x[mask],
            y[mask],
            self.beta_predict,
            restart_file='Be/benchmark_betafit_sub_domain.npy',
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
        x = self.temp[::self.temp_nslice][1:]
        x = numpy.concatenate([[0.0], x], axis=0)
        mask = x < self.temp_cutoff
        y = self.energy[::self.temp_nslice][1:]
        y = numpy.concatenate([[self.fci], y], axis=0)
        gpr = GPyDMQMC(
            x[mask],
            y[mask],
            self.temp_predict,
            restart_file='Be/benchmark_tempfit_sub_domain.npy',
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
