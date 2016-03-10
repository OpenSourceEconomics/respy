""" This module contains functionality that is shared between the solution
and simulation modules.
"""

# standard library
import numpy as np

import os

# project library
from robupy.constants import MISSING_FLOAT

''' Auxiliary functions
'''


def check_model_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks,
        shocks_cholesky):
    """ Check the integrity of all model parameters.
    """
    # Checks for all arguments
    args = [coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks,
            shocks_cholesky]
    for coeffs in args:
        assert (isinstance(coeffs, np.ndarray))
        assert (np.all(np.isfinite(coeffs)))
        assert (coeffs.dtype == 'float')

    # Checks for occupations
    assert (coeffs_a.size == 6)
    assert (coeffs_b.size == 6)
    assert (coeffs_edu.size == 3)
    assert (coeffs_home.size == 1)

    # Check Cholesky decomposition
    assert (shocks_cholesky.shape == (4, 4))
    aux = np.matmul(shocks_cholesky, shocks_cholesky.T)
    np.testing.assert_array_almost_equal(shocks, aux)

    # Checks shock matrix
    assert (shocks.shape == (4, 4))
    np.testing.assert_array_almost_equal(shocks, shocks.T)

    # Finishing
    return True


def distribute_model_paras(model_paras, is_debug):
    """ Distribute model parameters.
    """
    coeffs_a = model_paras['coeffs_a']
    coeffs_b = model_paras['coeffs_b']

    coeffs_edu = model_paras['coeffs_edu']
    coeffs_home = model_paras['coeffs_home']
    shocks = model_paras['shocks']
    shocks_cholesky = model_paras['shocks_cholesky']

    if is_debug:
        args = [coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks,
               shocks_cholesky]
        assert (check_model_parameters(*args))

    # Finishing
    return coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks, shocks_cholesky


def create_disturbances(num_periods, num_draws_emax, seed, is_debug, which,
        shocks_cholesky, is_ambiguous):
    """ Create the relevant set of disturbances. Handle special case of zero v
    variances as thi case is useful for hand-based testing. The disturbances
    are drawn from a standard normal distribution and transformed later in
    the code.
    """
    # Antibugging
    assert (which in ['emax', 'prob', 'sims'])

    # Control randomness by setting seed value
    np.random.seed(seed)

    # Draw random deviates from a standard normal distribution or read it in
    # from disk. The latter is available to allow for testing across
    # implementations.
    if is_debug and os.path.isfile('disturbances.txt'):
        disturbances = read_disturbances(num_periods, num_draws_emax)
    else:
        disturbances = np.random.multivariate_normal(np.zeros(4),
                        np.identity(4), (num_periods, num_draws_emax))

    # Deviates used for the Monte Carlo integration of the expected future
    # values during the solution of the model.
    if which == 'emax':
        # In the case of ambiguous world, the standard deviates are used in the
        # solution part of the program.
        for period in range(num_periods):
            disturbances[period, :, :] = \
                np.dot(shocks_cholesky, disturbances[period, :, :].T).T

        if not is_ambiguous:
            for period in range(num_periods):
                for j in [0, 1]:
                    disturbances[period, :, j] = \
                        np.exp(disturbances[period, :, j])

    # Deviates used for the Monte Carlo integration of the choice
    # probabilities for the construction of the criterion function.
    elif which == 'prob':
        # Standard deviates are further processed during the evaluation routine.
        disturbances = disturbances

    # Deviates for the simulation of a synthetic agent population.
    elif which == 'sims':
        # Standard deviates transformed to the distributions relevant for
        # the agents actual decision making as traversing the tree.
        for period in range(num_periods):
            disturbances[period, :, :] = \
                np.dot(shocks_cholesky, disturbances[period, :, :].T).T
            for j in [0, 1]:
                disturbances[period, :, j] = \
                    np.exp(disturbances[period, :, j])

    else:
        raise NotImplementedError

    # Finishing
    return disturbances


def replace_missing_values(argument):
    """ Replace missing value MISSING_FLOAT with NAN. Note that the output
    argument is
    of type float.
    """
    # Determine missing values
    is_missing = (argument == MISSING_FLOAT)

    # Transform to float array
    argument = np.asfarray(argument)

    # Replace missing values
    argument[is_missing] = np.nan

    # Finishing
    return argument


def read_disturbances(num_periods, num_draws_emax):
    """ Red the disturbances from disk. This is only used in the development
    process.
    """
    # Initialize containers
    disturbances_emax = np.tile(np.nan, (num_periods, num_draws_emax, 4))

    # Read and distribute disturbances
    disturbances = np.array(np.genfromtxt('disturbances.txt'), ndmin=2)
    for period in range(num_periods):
        lower = 0 + num_draws_emax * period
        upper = lower + num_draws_emax
        disturbances_emax[period, :, :] = disturbances[lower:upper, :]

    # Finishing
    return disturbances_emax


def check_dataset(data_frame, robupy_obj):
    """ This routine runs some consistency checks on the simulated data frame.
    """
    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')

    num_agents = robupy_obj.get_attr('num_agents')

    edu_max = robupy_obj.get_attr('edu_max')

    # Check dimension of data frame
    assert (data_frame.shape == (num_periods * num_agents, 8))

    # Check that there are the appropriate number of values
    for i in range(8):
        # This is not applicable for the wage observations
        if i == 3:
            continue
        # Check valid values
        assert (data_frame.count(axis=0)[i] == (num_periods * num_agents))

    # Check that the maximum number of values are valid
    for i, max_ in enumerate(
            [num_agents, num_periods, 4, num_periods, num_periods,
                num_periods,
                edu_max, 1]):
        # Agent and time index
        if i in [0, 1]:
            assert (data_frame.max(axis=0)[i] == (max_ - 1))
        # Choice observation
        if i == 2:
            assert (data_frame.max(axis=0)[i] <= max_)
        # Wage observations
        if i == 3:
            pass
        # Work experience A, B
        if i in [4, 5]:
            assert (data_frame.max(axis=0)[i] <= max_)
        # Education
        if i in [6]:
            assert (data_frame.max(axis=0)[i] <= max_)
        # Lagged Education
        if i in [7]:
            assert (data_frame.max(axis=0)[i] <= max_)

    # Each valid period indicator occurs as often as agents in the dataset.
    for period in range(num_periods):
        assert (data_frame[1].value_counts()[period] == num_agents)

    # Each valid agent indicator occurs as often as periods in the dataset.
    for agent in range(num_agents):
        assert (data_frame[0].value_counts()[agent] == num_periods)

    # Check valid values of wage observations
    for count in range(num_agents * num_periods):
        is_working = (data_frame[2][count] in [1, 2])
        if is_working:
            assert (np.isfinite(data_frame[3][count]))
            assert (data_frame[3][count] > 0.00)
        else:
            assert (np.isfinite(data_frame[3][count]) == False)


def check_optimization_parameters(x):
    """ Check optimization parameters.
    """
    # Perform checks
    assert (isinstance(x, np.ndarray))
    assert (x.dtype == np.float)
    assert (x.shape == (26,))
    assert (np.all(np.isfinite(x)))

    # Finishing
    return True


def opt_get_optim_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks, shocks_cholesky, is_debug):
    """ Get optimization parameters.
    """
    # Checks
    if is_debug:
        args = [coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks,
               shocks_cholesky]
        assert check_model_parameters(*args)

    # Initialize container
    x = np.tile(np.nan, 26)

    # Occupation A
    x[0:6] = coeffs_a

    # Occupation B
    x[6:12] = coeffs_b

    # Education
    x[12:15] = coeffs_edu

    # Home
    x[15:16] = coeffs_home

    # Shocks
    x[16:20] = shocks_cholesky[0:4, 0]
    x[20:23] = shocks_cholesky[1:4, 1]
    x[23:25] = shocks_cholesky[2:4, 2]
    x[25:26] = shocks_cholesky[3:4, 3]

    # Checks
    if is_debug:
        check_optimization_parameters(x)

    # Finishing
    return x


def opt_get_model_parameters(x, is_debug):
    """ Update parameter values. The np.array type is maintained.
    """
    # Checks
    if is_debug:
        check_optimization_parameters(x)

    # Occupation A
    coeffs_a = x[0:6]

    # Occupation B
    coeffs_b = x[6:12]

    # Education
    coeffs_edu = x[12:15]

    # Home
    coeffs_home = x[15:16]

    # Cholesky
    shocks_cholesky = np.tile(0.0, (4, 4))

    shocks_cholesky[0:4, 0] = x[16:20]
    shocks_cholesky[1:4, 1] = x[20:23]
    shocks_cholesky[2:4, 2] = x[23:25]
    shocks_cholesky[3:4, 3] = x[25]

    # Shocks
    shocks = np.matmul(shocks_cholesky, shocks_cholesky.T)

    # Checks
    if is_debug:
        args = [coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks,
               shocks_cholesky]
        assert check_model_parameters(*args)

    # Finishing
    return coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks, shocks_cholesky
