""" This module contains functionality that is shared between the solution
and simulation modules.
"""

# standard library
import numpy as np
import os

# project library
from robupy.constants import MISSING_FLOAT


def check_model_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks, eps_cholesky):
    """ Check the integrity of all model parameters.
    """
    # Checks for all arguments
    args = [coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks, eps_cholesky]
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
    assert (eps_cholesky.shape == (4, 4))
    aux = np.matmul(eps_cholesky, eps_cholesky.T)
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
    eps_cholesky = model_paras['eps_cholesky']

    if is_debug:
        args = coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks, eps_cholesky
        assert (check_model_parameters(*args))

    # Finishing
    return coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks, eps_cholesky


def create_disturbances(num_draws, seed, eps_cholesky, is_ambiguous,
        num_periods, is_debug, which):
    """ Create disturbances.  Handle special case of zero variances as this
    case is useful for hand-based testing. The disturbances are drawn from a
    standard normal distribution and transformed later in the code.
    """
    # Antibugging.
    assert (which in ['simulation', 'estimation', 'solution'])

    # Auxiliary objects
    is_simulation = (which == 'simulation')

    is_estimation = (which == 'estimation')

    # Initialize container
    periods_eps_relevant = np.tile(MISSING_FLOAT, (num_periods, num_draws, 4))

    # Draw standard normal deviates used to simulate the likelihood function.
    if is_estimation:
        np.random.seed(seed)
        standard_deviates = np.random.multivariate_normal(np.zeros(4),
            np.identity(4), (num_periods, num_draws))
        return standard_deviates

    # This allows to use the same random disturbances across the different
    # implementations of the mode, including the RESTUD program. Otherwise,
    # we draw a new set of standard deviations
    if is_debug and os.path.isfile('disturbances.txt'):
        standard_deviates = read_disturbances(num_periods, num_draws)
        standard_deviates = standard_deviates[:num_periods, :num_draws, :]
    else:
        np.random.seed(seed)
        standard_deviates = np.random.multivariate_normal(np.zeros(4),
            np.identity(4), (num_periods, num_draws))

    # In the case of ambiguous world, the standard deviates are used in the
    # solution part of the program.
    if is_ambiguous and not is_simulation:
        for period in range(num_periods):
            periods_eps_relevant[period, :, :] = np.dot(eps_cholesky,
                    standard_deviates[period, :, :].T).T
    else:
        # Transform disturbances to relevant distribution
        for period in range(num_periods):
            periods_eps_relevant[period, :, :] = np.dot(eps_cholesky,
                    standard_deviates[period, :, :].T).T
            for j in [0, 1]:
                periods_eps_relevant[period, :, j] = np.exp(periods_eps_relevant[
                                                          period, :, j])

    # Finishing
    return periods_eps_relevant


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


def read_disturbances(num_periods, num_draws):
    """ Red the disturbances from disk. This is only used in the development
    process.
    """
    # Initialize containers
    periods_eps_relevant = np.tile(np.nan, (num_periods, num_draws, 4))

    # Read and distribute disturbances
    disturbances = np.array(np.genfromtxt('disturbances.txt'), ndmin=2)
    for period in range(num_periods):
        lower = 0 + num_draws * period
        upper = lower + num_draws
        periods_eps_relevant[period, :, :] = disturbances[lower:upper, :]

    # Finishing
    return periods_eps_relevant


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

