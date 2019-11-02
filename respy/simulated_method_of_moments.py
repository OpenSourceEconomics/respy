"""Estimate models with the simulated method of moments.

The simulated method of moments is developed by [1], [2], and [3] and an estimation
technique where the distance between the moments of the actual data and the moments
implied by the model parameters is minimized.

References
----------

.. [1] McFadden, D. (1989). A method of simulated moments for estimation of discrete
       response models without numerical integration. Econometrica: Journal of the
       Econometric Society, 995-1026.
.. [2] Lee, B. S., & Ingram, B. F. (1991). Simulation estimation of time-series models.
       Journal of Econometrics, 47(2-3), 197-205.
.. [3] Duffie, D., & Singleton, K. (1993). Simulated Moments Estimation of Markov Models
       of Asset Prices. Econometrica, 61(4), 929-952.

"""
import functools

import numpy as np

from respy.pre_processing.model_processing import process_params_and_options
from respy.simulate import get_simulate_func


def get_smm_func(params, options, moments, calc_moments, weighting_matrix=None):
    """Get the criterion function for estimation with SMM.

    Return a version of the :func:`smm` function where all arguments except the
    parameter vector are fixed with :func:`functools.partial`. Thus the function can be
    directly passed into an optimizer or a function for taking numerical derivatives.

    Parameters
    ----------
    params : pandas.DataFrame
        DataFrame containing model parameters.
    options : dict
        Dictionary containing model options.
    moments : np.ndarray
        Array with shape (n_moments,) containing the moments of the actual data.
    calc_moments : callable
        Function to compute the moments from the simulated data.
    weighting_matrix : np.ndarray
        Array with shape (n_moments, n_moments) which is used to weight the squared
        errors of moments.

    Returns
    -------
    smm_function : :func:`smm`
        Criterion function where all arguments except the parameter vector are set.

    """
    optim_paras, options = process_params_and_options(params, options)

    simulate = get_simulate_func(params, options)

    if weighting_matrix is None:
        weighting_matrix = np.eye(len(moments))

    smm_function = functools.partial(
        smm,
        simulate=simulate,
        moments=moments,
        weighting_matrix=weighting_matrix,
        calc_moments=calc_moments,
    )

    return smm_function


def smm(params, simulate, moments, weighting_matrix, calc_moments):
    """Criterion function for the estimation with SMM.

    This function calculates the sum of weighted squared errors of moments.

    Parameters
    ----------
    params : pandas.Series
        Parameter Series.
    simulate : :func:`~respy.simulate.simulate`
        Function to simulate a new data set for a given set of parameters.
    moments : np.ndarray
        Array with shape (n_moments,) containing the moments from the actual data.
    weighting_matrix : np.ndarray
        Array with shape (n_moments, n_moments) which is used to weight the squared
        errors of moments.
    calc_moments : callable
        Function to compute the moments from the simulated data.

    """
    df = simulate(params)

    estimated_moments = calc_moments(df)

    estimated_moments = np.clip(estimated_moments, -1e250, 1e250)
    estimated_moments[np.isnan(estimated_moments)] = 0

    moments_error = estimated_moments - moments

    weighted_sum_squared_errors = moments_error @ weighting_matrix @ moments_error

    weighted_sum_squared_errors = np.clip(weighted_sum_squared_errors, -1e300, 1e300)

    with open("logging.txt", "a+") as file:
        file.write(f"Error: {weighted_sum_squared_errors}\n")
        file.write(params.to_string())
        file.write("\n")
        file.write("-" * 88)
        file.write("\n\n")

    return weighted_sum_squared_errors
