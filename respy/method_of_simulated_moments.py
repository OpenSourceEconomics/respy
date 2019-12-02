"""Estimate models with the method of simulated moments (MSM).

The method of simulated moments is developed by [1], [2], and [3] and an estimation
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


def get_msm_func(params, options, moments, calc_moments, weighting_matrix=None):
    """Get the criterion function for estimation with MSM.

    Return a version of the :func:`msm` function where all arguments except the
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
    msm_function : :func:`msm`
        Criterion function where all arguments except the parameter vector are set.

    """
    optim_paras, options = process_params_and_options(params, options)

    simulate = get_simulate_func(params, options)

    if weighting_matrix is None:
        weighting_matrix = np.eye(len(moments))

    msm_function = functools.partial(
        msm,
        simulate=simulate,
        moments=moments,
        weighting_matrix=weighting_matrix,
        calc_moments=calc_moments,
    )

    return msm_function


def msm(params, simulate, moments, weighting_matrix, calc_moments):
    """Criterion function for the estimation with MSM.

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

    #Get rid of nan values
    estimated_moments = estimated_moments.dropna()

    #Get index that is not contained
    missing_moments = []
    for x in moments.index:
        if x not in estimated_moments.index:
            missing_moments.append(x)

    #Assign new values
    estimated_moments[missing_moments] = 0



    moments_error = estimated_moments[moments.index].as_numpy() - moments.as_numpy()

    return moments_error @ weighting_matrix @ moments_error
