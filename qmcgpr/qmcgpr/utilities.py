#!/usr/bin/env python

import numpy

from typing import TypeVar, Tuple

Array = TypeVar('numpy.ndarray')


def form_combined_dataset(
            crossover: float,
            temperature: Array,
            temperature_data: Array,
            beta: Array,
            beta_data: Array,
        ) -> Tuple[Array, Array]:
    ''' Generate a single x-axis and y-axis array pair given the
    data for the temperature domain and range, the beta domain and range,
    and the crossover where the two should contribute data from. The crossover
    is given in temperature. The data from each domain/range pair is taken as:
        f(T) for T < crossover and f(beta) for 1/beta >= crossover.
    When we say temperature it is implied we are referring to data generated
    from a GPR model trained on the temperature domain.
    Similarly, beta refers to data resulting from training on the beta domain.
    And the temperature and beta domain are related like:
        T = 1/beta.

    Note, we sort the resulting product based on the returned x-axis.

    Parameters
    ----------
    crossover : float
        The crossover temperature (in Hartree) where the two data sets should
        be combined on.
    temperature : :class:`numpy.ndarray`
        The temperatures for the temperature data set.
    temperature_data : :class:`numpy.ndarray`
        The data for the temperature data set.
    beta : :class:`numpy.ndarray`
        The beta for the beta data set.
    beta_data : :class:`numpy.ndarray`
        The data for the beta data set.

    Returns
    -------
    X : :class:`numpy.ndarray`
        The x-axis temperatures constructed by combining the temperature and
        beta data sets based on the crossover temperature.
    Y : :class:`numpy.ndarray`
        The y-axis quantities constructed by combining the data for the
        temperature and beta data sets based on the crossover temperature.
    '''
    temperature_from_beta = numpy.divide(1.0, beta)
    temperature_from_beta[beta == 0.0] = numpy.inf
    temperature_from_beta[beta == numpy.inf] = 0.0

    beta_mask = temperature_from_beta >= crossover

    temperature_mask = temperature < crossover

    X = numpy.concatenate(
        [temperature_from_beta[beta_mask], temperature[temperature_mask]],
        axis=0,
    )
    Y = numpy.concatenate(
        [beta_data[beta_mask], temperature_data[temperature_mask]],
        axis=0,
    )

    sorted_indices = numpy.argsort(X)

    X = X[sorted_indices]
    Y = Y[sorted_indices]

    return X, Y


def numerical_entropy(
            temperature: Array,
            specific_heat: Array,
        ) -> Tuple[Array, Array]:
    ''' Numerically integrate the electronic specific heat capacity to
    calculate an estimate for the electronic entropy. The integral being
    numerically sampled is:
        S(T) = integral_{0}^{T} C_{V}(T') 1/T' dT'
    for more details see:
        https://doi.org/10.1063/5.0150702.

    Note, the return temperature and numerically approximated entropy
    are sorted by the returned temperature from smallest to largest.
    Furthermore, the NumPy trapz method does not like non-finite values
    so these are masked away.

    Parameters
    ----------
    temperature : :class:`numpy.ndarray`
        The temperature corresponding to the specific heats.
    specific_heat : :class:`numpy.ndarray`
        The electronic specific heats to numerically integrate over.

    Returns
    -------
    X : :class:`numpy.ndarray`
        The temperatures for the integration, sorted from smallest to largest.
    entropy : :class:`numpy.ndarray`
        The numerically estimated electronic entropy for the given
        sorted temperatures in X.
    '''
    nonzero_mask = temperature != 0.0

    finite_mask = numpy.logical_and(
        numpy.isfinite(temperature),
        numpy.isfinite(specific_heat),
    )

    mask = numpy.logical_and(
        nonzero_mask,
        finite_mask,
    )

    masked_temperature = temperature[mask]
    masked_specific_heat = specific_heat[mask]

    ratio = numpy.divide(
        masked_specific_heat,
        masked_temperature,
    )

    sorted_indices = numpy.argsort(masked_temperature)

    X = masked_temperature[sorted_indices]
    Y = ratio[sorted_indices]

    X = numpy.concatenate([[0.0], X], axis=0)
    Y = numpy.concatenate([[0.0], Y], axis=0)

    entropy = numpy.zeros(X.shape)

    for index in range(X.shape[0]):
        entropy[index] = numpy.trapz(Y[:index], x=X[:index])

    return X, entropy
