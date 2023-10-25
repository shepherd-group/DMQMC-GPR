#!/usr/bin/env python

import os
import numpy
import pandas

from typing import TypeVar, List, Tuple

Array = TypeVar('numpy.ndarray')
Dataframe = TypeVar('pandas.core.frame.DataFrame')


class DataDMQMC:
    ''' A class to read in HANDE DMQMC data and prepare for fitting.

    Attributes
    ----------
    outputs : list of str
        The output filenames for HANDE-QMC DMQMC data.
    taus : list of float
        The time step corresponding to each individual beta loop read in.
    dataframes : list of :class:`pandas.DataFrame`
        The data for individual beta loops, corresponds to the tau for the
        same index in taus.

    Methods
    -------
    get_data_for_training(key, nslice, target_beta)
        Prepares an x and y array for performing GPR.
    _read_dmqmc()
        Read in the provided outputs and store the data.
    '''
    def __init__(
                self,
                outputs: List[str],
            ) -> None:
        ''' Take in output files and store the data so that the
        GPR training data can be generated.

        Parameters
        ----------
        outputs : list of str
            The output files which contain the individual beta loops to read
            in and store the data for.
        '''
        self.outputs = outputs
        self._read_dmqmc()

    def _read_dmqmc(self) -> None:
        ''' Read and store the data which can be processed in the
        training data preparer to use with GPR.

        Raises
        ------
        RuntimeError
            If the output file does not exist.
        RuntimeError
            If the taus do not agree between the simulations.
        '''
        self.taus = []
        self.dataframes = []

        for output in self.outputs:
            if not os.path.isfile(output):
                raise RuntimeError('Failed to find HANDE output file:\n'
                                   f'    {output}')

            taus, dataframes = read_hande_dmqmc(output)

            for tau, dataframe in zip(taus, dataframes):
                dataframe['Beta'] = dataframe['iterations']*tau
                dataframe['E'] = dataframe[r'\sum\rho_{ij}H_{ji}']
                dataframe['E'] /= dataframe['Trace']
                self.taus.append(tau)
                self.dataframes.append(dataframe)

        if numpy.unique(self.taus).shape != (1,):
            print(self.taus)
            raise RuntimeError('The time-step does not match for the '
                               'provided simulations!')

    def get_data_for_training(
                self,
                key: str = 'E',
                nslice: int = 1,
                target_beta: float = 0.0,
            ) -> Tuple[Array, Array]:
        ''' Generate the x and y data sets for training using the
        individual beta loops read in.

        Parameters
        ----------
        key : str, default = "E"
            The key to index the y-axis values.
        nslice : int, default = 1
            An integer to resample the individual data sets by.
        target_beta : float, default = 0.0
            A target beta to exclude data for beta values below this.

        Returns
        -------
        x : :class:`numpy.ndarray`
            The x-axis data for performing GPR, corresponds to the inverse
            temperature for a DMQMC simulations.
        y : :class:`numpy.ndarray`
            The y-axis data for performing GPR.
        '''
        sliced_dataframes = [df[::nslice] for df in self.dataframes]

        dataframe = pandas.concat(sliced_dataframes)
        dataframe = dataframe[dataframe['Beta'] >= target_beta]
        dataframe = dataframe.reset_index(drop=True)

        x = dataframe['Beta'].values.flatten()
        y = dataframe[key].values.flatten()

        return x, y


def read_hande_dmqmc(output: str) -> Tuple[List[float], List[Dataframe]]:
    ''' A very basic function for reading in HANDE DMQMC data.
    The data is returned in a dataframe.

    Parameters
    ----------
    output : str
        The HANDE output file.

    Returns
    -------
    taus : list of float
        The time step for the simulations.
    dataframes : list of :class:`pandas.DataFrame`
        The dataframes with the simulation data.
    '''
    taus = []
    dataframes = []
    use_line = False

    with open(output, 'r') as stream:
        for line in stream:
            line_data = line.split()
            if '#     iterations ' in line:
                use_line = True
                columns = _get_dmqmc_header_columns(line)
                df = {k: [] for k in columns}
            elif '"tau":' in line:
                tau = float(line_data[-1].replace(',', ''))
            elif '# Resetting beta... Beta loop =' in line:
                taus.append(tau)
                dataframes.append(pandas.DataFrame(df))
                df = {k: [] for k in columns}
            elif '#' in line:
                continue
            elif use_line and len(line_data) == 0:
                use_line = False
                taus.append(tau)
                dataframes.append(pandas.DataFrame(df))
                columns = _get_dmqmc_header_columns(line)
                df = {k: [] for k in columns}
            elif use_line:
                for k, v in zip(columns, line_data):
                    if '.' in v or 'E' in v:
                        df[k].append(float(v))
                    else:
                        df[k].append(int(v))

    return taus, dataframes


def _get_dmqmc_header_columns(header: str) -> List[str]:
    ''' Take in a header string for a HANDE DMQMC beta loop simulation
    and extract the column names and return as a list of strings.

    Parameters
    ----------
    header : str
        The header for the beta loop.

    Returns
    -------
    columns : list of str
        The list of column headers.
    '''
    line = header.replace(' ', '~')
    for Ntilde in range(2, 31):
        line = line.replace('~'*Ntilde, ' '*Ntilde)
    columns = line.split()[1:-1]
    for i, c in enumerate(columns):
        if c[0] == '~':
            columns[i] = c[1:]
        columns[i] = columns[i].replace('~', ' ')
    return columns
