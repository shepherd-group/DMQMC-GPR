#!/usr/bin/env python

from os import path
from numpy import allclose


class TestDataTolerance:
    r''' A simple class for performing the data comparisons for
    the simple tests.

    Attributes
    ----------
    __test__ : bool = False
        For pytest only, indicates this class is not tested.
    rtol : float = 1e-05
        The rtol value used when calling the NumPy allclose function.
        This can be altered using the simple_test_tolerance.txt file.
    atol : float = 1e-08
        The atol value used when calling the NumPy allclose function.
        This can be altered using the simple_test_tolerance.txt file.

    Methods
    -------
    get_tolerance()
        Called during initalization, checks for and reads in the
        simple_test_tolerance.txt file to set the rtol and atol parameters.
    extract_tol(line)
        Called from get_tolerance to get the specific value for rtol
        or the atol parameters.
    __repr__()
        Generate a simple report for the current rtol and atol values.
    __getitem__(a, b, m)
        Perform the comparison on the sub-arrays for arrays a and b, generated
        with mask m, using the NumPy allclose function.
    '''
    __test__: bool = False
    rtol: float = 1e-05
    atol: float = 1e-08

    def __init__(self) -> None:
        r''' Attempt to set the tolerances from the
        simple_test_tolerance.txt file.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''
        self.get_tolerances()
        return

    def get_tolerances(self) -> None:
        r''' Read in the parameters within the
        simple_test_tolerance.txt file if it exists.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''
        rtol_line_term = 'rtol:'
        atol_line_term = 'atol:'
        tolerance_file = 'simple_test_tolerance.txt'

        if not path.isfile(tolerance_file):
            return

        with open(tolerance_file, 'rt') as stream:
            for line in stream:
                if rtol_line_term in line:
                    self.rtol = self.extract_tol(line)
                elif atol_line_term in line:
                    self.atol = self.extract_tol(line)

        return

    @staticmethod
    def extract_tol(line: str) -> float:
        r''' Extract the rtol or atol values defined in the
        simple_test_tolerance.txt file.

        Parameters
        ----------
        line : str
            The line we are attemping to extract the value from.

        Returns
        -------
        tol : float
            The value for rtol or atol found in the line.
        '''
        return float(line.split(':')[-1])

    def __repr__(self) -> str:
        r''' Generates a simple report with the values assigned to rtol
        and atol. For checking the values were set properly when using the
        simple_test_tolerance.txt file.

        Parameters
        ----------
        None.

        Returns
        -------
        report : str
            The report message with the values of rtol and atol.
        '''
        report = (
            '=== test tolerance parameters ===\n'
            f' rtol: {self.rtol}\n'
            f' atol: {self.atol}\n'
            '================================='
        )
        return report

    def __getitem__(self, test_arrays: tuple) -> bool:
        r''' Perform an AB and BA test using the NumPy allclose function
        to determine the agreement between data. If the data are in agreement
        in both the AB and BA ordered comparison return True for agreement,
        otherwise return False for disagreement.

        Parameters
        ----------
        a : :class:`numpy.ndarray`
            The first array to test.
        b : :class:`numpy.ndarray`
            The second array to test.
        m : :class:`numpy.ndarray`
            The mask array applied to arrays a and b.

        Returns
        -------
        isclose : bool
            True if the AB and BA tests pass given the rtol and atol values
            used to call allclose.
        '''
        a, b, m = test_arrays

        ab_isclose = allclose(
            a[m],
            b[m],
            rtol=self.rtol,
            atol=self.atol,
            equal_nan=True,
        )

        ba_isclose = allclose(
            b[m],
            a[m],
            rtol=self.rtol,
            atol=self.atol,
            equal_nan=True,
        )

        isclose = ab_isclose and ba_isclose

        return isclose


if __name__ == '__main__':
    # A quick report for the simple test parameters.
    print(TestDataTolerance())
    #help(TestDataTolerance)
