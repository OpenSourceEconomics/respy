import os

import numpy as np

from robupy.shared.constants import HUGE_FLOAT, MISSING_FLOAT


def get_total_value(period, num_periods, delta, payoffs_systematic, draws,
        edu_max, edu_start, mapping_state_idx, periods_emax, k, states_all):
    """ Get total value of all possible states.
    """
    # Initialize containers
    payoffs_ex_post = np.tile(np.nan, 4)

    # Calculate ex post payoffs
    for j in [0, 1]:
        payoffs_ex_post[j] = payoffs_systematic[j] * draws[j]

    for j in [2, 3]:
        payoffs_ex_post[j] = payoffs_systematic[j] + draws[j]

    # Get future values
    if period != (num_periods - 1):
        payoffs_future = _get_future_payoffs(edu_max, edu_start,
            mapping_state_idx, period, periods_emax, k, states_all)
    else:
        payoffs_future = np.tile(0.0, 4)

    # Calculate total utilities
    total_payoffs = payoffs_ex_post + delta * payoffs_future

    # This is required to ensure that the agent does not choose any
    # inadmissible states.
    if payoffs_future[2] == -HUGE_FLOAT:
        total_payoffs[2] = -HUGE_FLOAT

    # Finishing
    return total_payoffs, payoffs_ex_post, payoffs_future


def _get_future_payoffs(edu_max, edu_start, mapping_state_idx, period,
        periods_emax, k, states_all):
    """ Get future payoffs for additional choices.
    """
    # Distribute state space
    exp_a, exp_b, edu, edu_lagged = states_all[period, k, :]

    # Future utilities
    payoffs_future = np.tile(np.nan, 4)

    # Working in occupation A
    future_idx = mapping_state_idx[period + 1, exp_a + 1, exp_b, edu, 0]
    payoffs_future[0] = periods_emax[period + 1, future_idx]

    # Working in occupation B
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b + 1, edu, 0]
    payoffs_future[1] = periods_emax[period + 1, future_idx]

    # Increasing schooling. Note that adding an additional year
    # of schooling is only possible for those that have strictly
    # less than the maximum level of additional education allowed.
    if edu < edu_max - edu_start:
        future_idx = mapping_state_idx[period + 1, exp_a, exp_b, edu + 1, 1]
        payoffs_future[2] = periods_emax[period + 1, future_idx]
    else:
        payoffs_future[2] = -HUGE_FLOAT

    # Staying at home
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b, edu, 0]
    payoffs_future[3] = periods_emax[period + 1, future_idx]

    # Finishing
    return payoffs_future


def create_draws(num_periods, num_draws_emax, seed, is_debug, which,
        shocks_cholesky):
    """ Create the relevant set of draws. Handle special case of zero v
    variances as thi case is useful for hand-based testing. The draws
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
    if is_debug and os.path.isfile('draws.txt'):
        draws = read_draws(num_periods, num_draws_emax)
    else:
        draws = np.random.multivariate_normal(np.zeros(4),
                        np.identity(4), (num_periods, num_draws_emax))
    # Standard normal deviates used for the Monte Carlo integration of the
    # expected future values in the solution step. Also, standard normal
    # deviates for the Monte Carlo integration of the choice probabilities in
    # the evaluation step.
    if which in ['prob', 'emax']:
        # Standard deviates are further processed during the evaluation routine.
        draws = draws

    # Deviates for the simulation of a synthetic agent population.
    elif which == 'sims':
        # Standard deviates transformed to the distributions relevant for
        # the agents actual decision making as traversing the tree.
        for period in range(num_periods):
            draws[period, :, :] = \
                np.dot(shocks_cholesky, draws[period, :, :].T).T
            for j in [0, 1]:
                draws[period, :, j] = np.exp(draws[period, :, j])

    else:
        raise NotImplementedError

    # Finishing
    return draws


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


def read_draws(num_periods, num_draws_emax):
    """ Red the draws from disk. This is only used in the development
    process.
    """
    # Initialize containers
    periods_draws_emax = np.tile(np.nan, (num_periods, num_draws_emax, 4))

    # Read and distribute draws
    draws = np.array(np.genfromtxt('draws.txt'), ndmin=2)
    for period in range(num_periods):
        lower = 0 + num_draws_emax * period
        upper = lower + num_draws_emax
        periods_draws_emax[period, :, :] = draws[lower:upper, :]

    # Finishing
    return periods_draws_emax


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


def distribute_model_paras(model_paras, is_debug):
    """ Distribute model parameters.
    """
    coeffs_a = model_paras['coeffs_a']
    coeffs_b = model_paras['coeffs_b']

    coeffs_edu = model_paras['coeffs_edu']
    coeffs_home = model_paras['coeffs_home']
    shocks_cov = model_paras['shocks_cov']
    shocks_cholesky = model_paras['shocks_cholesky']

    if is_debug:
        args = [coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
               shocks_cholesky]
        assert (check_model_parameters(*args))

    # Collect arguments
    args = [coeffs_a, coeffs_b, coeffs_edu, coeffs_home]
    args += [shocks_cov, shocks_cholesky]

    # Finishing
    return args


def check_model_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cov, shocks_cholesky):
    """ Check the integrity of all model parameters.
    """
    # Checks for all arguments
    args = [coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
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
    np.testing.assert_array_almost_equal(shocks_cov, aux)

    # Checks shock matrix
    assert (shocks_cov.shape == (4, 4))
    np.testing.assert_array_almost_equal(shocks_cov, shocks_cov.T)

    # Finishing
    return True