# standard library
import numpy as np
import os

# project library
from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.shared.shared_constants import HUGE_FLOAT


def cut_dataset(respy_obj, data_frame):
    """ Cut the dataset down to only contain the agents used in the estimation.
    """
    # Distribute class attributes
    num_agents_est = respy_obj.get_attr('num_agents_est')
    num_periods = respy_obj.get_attr('num_periods')

    # Slicing data frame
    data_frame = data_frame.iloc[:(num_agents_est * num_periods), :]

    # Finishing
    return data_frame


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
    return total_payoffs


def _get_future_payoffs(edu_max, edu_start, mapping_state_idx, period,
        periods_emax, k, states_all):
    """ Get future payoffs for additional choices.
    """
    # Distribute state space
    exp_a, exp_b, edu, _ = states_all[period, k, :]

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


def create_draws(num_periods, num_draws, seed, is_debug):
    """ Create the relevant set of draws. Handle special case of zero v
    variances as thi case is useful for hand-based testing. The draws
    are drawn from a standard normal distribution and transformed later in
    the code.
    """
    # Control randomness by setting seed value
    np.random.seed(seed)

    # Draw random deviates from a standard normal distribution or read it in
    # from disk. The latter is available to allow for testing across
    # implementations.
    if is_debug and os.path.exists('draws.txt'):
        draws = read_draws(num_periods, num_draws)
    else:
        draws = np.random.multivariate_normal(np.zeros(4),
                        np.identity(4), (num_periods, num_draws))
    # Finishing
    return draws


def add_solution(respy_obj, store, periods_payoffs_systematic,
        states_number_period, mapping_state_idx, periods_emax, states_all):
    """ Add solution to class instance.
    """
    respy_obj.unlock()

    respy_obj.set_attr('periods_payoffs_systematic', periods_payoffs_systematic)

    respy_obj.set_attr('states_number_period', states_number_period)

    respy_obj.set_attr('mapping_state_idx', mapping_state_idx)

    respy_obj.set_attr('periods_emax', periods_emax)

    respy_obj.set_attr('states_all', states_all)

    respy_obj.set_attr('is_solved', True)

    respy_obj.lock()

    # Store object to file
    if store:
        respy_obj.store('solution.respy.pkl')

    # Finishing
    return respy_obj


def check_dataset(data_frame, respy_obj, which):
    """ This routine runs some consistency checks on the simulated data frame.
    """
    # Distribute class attributes
    num_periods = respy_obj.get_attr('num_periods')

    edu_max = respy_obj.get_attr('edu_max')

    if which == 'est':
        num_agents = respy_obj.get_attr('num_agents_est')
    elif which == 'sim':
        num_agents = respy_obj.get_attr('num_agents_sim')
    else:
        raise AssertionError
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


def dist_model_paras(model_paras, is_debug):
    """ Distribute model parameters.
    """
    coeffs_a = model_paras['coeffs_a']
    coeffs_b = model_paras['coeffs_b']

    coeffs_edu = model_paras['coeffs_edu']
    coeffs_home = model_paras['coeffs_home']
    shocks_cov = model_paras['shocks_cov']

    # Construct cholesky decomposition
    if np.count_nonzero(shocks_cov) == 0:
        shocks_cholesky = np.zeros((4, 4))
    else:
        shocks_cholesky = np.linalg.cholesky(shocks_cov)

    if is_debug:
        args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            shocks_cholesky)
        assert (check_model_parameters(*args))

    # Collect arguments
    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home)
    args += (shocks_cov, shocks_cholesky)

    # Finishing
    return args


def replace_missing_values(arguments):
    """ Replace missing value MISSING_FLOAT with NAN. Note that the output
    argument is of type float in the case missing values are found.
    """
    # Antibugging
    assert (isinstance(arguments, tuple) or isinstance(arguments, np.ndarray))

    if isinstance(arguments, np.ndarray):
        arguments = (arguments, )

    rslt = tuple()

    for argument in arguments:
        # Transform to float array to evaluate missing values.
        argument_internal = np.asfarray(argument)

        # Determine missing values
        is_missing = (argument_internal == MISSING_FLOAT)
        if np.any(is_missing):
            # Replace missing values
            argument = np.asfarray(argument)
            argument[is_missing] = np.nan

        rslt += (argument,)

    # Aligning interface.
    if len(rslt) == 1:
        rslt = rslt[0]

    # Finishing
    return rslt


def check_model_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cov, shocks_cholesky):
    """ Check the integrity of all model parameters.
    """
    # Checks for all arguments
    args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            shocks_cholesky)
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


def dist_class_attributes(respy_obj, *args):
    """ This function distributes a host of class attributes.
    """
    # Initialize container
    ret = []

    # Process requests
    for arg in args:
        ret.append(respy_obj.get_attr(arg))

    # Finishing
    return ret


def read_draws(num_periods, num_draws):
    """ Red the draws from disk. This is only used in the development
    process.
    """
    # Initialize containers
    periods_draws = np.tile(np.nan, (num_periods, num_draws, 4))

    # Read and distribute draws
    draws = np.array(np.genfromtxt('draws.txt'), ndmin=2)
    for period in range(num_periods):
        lower = 0 + num_draws * period
        upper = lower + num_draws
        periods_draws[period, :, :] = draws[lower:upper, :]

    # Finishing
    return periods_draws


def transform_disturbances(draws, shocks_cholesky):
    """ Transform the standard normal deviates to the relevant distribution.
    """
    # Transfer draws to relevant distribution
    draws_transformed = draws.copy()
    draws_transformed = np.dot(shocks_cholesky, draws_transformed.T).T

    for j in [0, 1]:
        draws_transformed[:, j] = \
            np.clip(np.exp(draws_transformed[:, j]), 0.0, HUGE_FLOAT)

    # Finishing
    return draws_transformed


def print_init_dict(dict_, file_name='test.respy.ini'):
    """ Print initialization dictionary to file. The different formatting
    makes the file rather involved. The resulting initialization files are
    read by PYTHON and FORTRAN routines. Thus, the formatting with respect to
    the number of decimal places is rather small.
    """
    # Antibugging.
    assert (isinstance(dict_, dict))

    paras_fixed = dict_['OCCUPATION A']['fixed'][:]
    paras_fixed += dict_['OCCUPATION B']['fixed'][:]
    paras_fixed += dict_['EDUCATION']['fixed'][:]
    paras_fixed += dict_['HOME']['fixed'][:]
    paras_fixed += [dict_['SHOCKS']['fixed'][0]][:]

    str_optim = '{0:<10} {1:20.4f} {2:>5} \n'

    # Construct labels. This ensures that the initialization files alway look
    # identical.
    labels = ['BASICS', 'AMBIGUITY', 'OCCUPATION A', 'OCCUPATION B']
    labels += ['EDUCATION', 'HOME', 'SHOCKS',  'SOLUTION']
    labels += ['SIMULATION', 'ESTIMATION', 'PROGRAM', 'INTERPOLATION']
    labels += ['SCIPY-BFGS', 'SCIPY-POWELL']

    # Create initialization.
    with open(file_name, 'w') as file_:

        for flag in labels:
            if flag in ['BASICS']:

                file_.write('BASICS \n\n')

                str_ = '{0:<10} {1:>20} \n'
                file_.write(str_.format('periods', dict_[flag]['periods']))

                str_ = '{0:<10} {1:20.4f} \n'
                file_.write(str_.format('delta', dict_[flag]['delta']))

                file_.write('\n')

            if flag in ['HOME']:

                file_.write(flag.upper() + '\n\n')

                val = dict_['HOME']['coeffs'][0]
                line = format_opt_parameters(val, 15, paras_fixed)
                file_.write(str_optim.format(*line))

                file_.write('\n')

            if flag in ['SOLUTION', 'SIMULATION', 'PROGRAM', 'INTERPOLATION',
                        'ESTIMATION']:

                file_.write(flag.upper() + '\n\n')

                for keys_ in dict_[flag]:

                    if keys_ in ['tau']:
                        str_ = '{0:<10} {1:20.0f} \n'
                        file_.write(str_.format(keys_, dict_[flag][keys_]))
                    else:
                        str_ = '{0:<10} {1:>20} \n'
                        file_.write(str_.format(keys_, str(dict_[flag][keys_])))

                file_.write('\n')

            if flag in ['SHOCKS']:

                # Type conversion
                file_.write(flag.upper() + '\n\n')

                for i in range(10):
                    val = dict_['SHOCKS']['coeffs'][i]
                    line = format_opt_parameters(val, 16, paras_fixed)
                    file_.write(str_optim.format(*line))
                file_.write('\n')

            if flag in ['EDUCATION']:

                file_.write(flag.upper() + '\n\n')

                val = dict_['EDUCATION']['coeffs'][0]
                line = format_opt_parameters(val, 12, paras_fixed)
                file_.write(str_optim.format(*line))

                val = dict_['EDUCATION']['coeffs'][1]
                line = format_opt_parameters(val, 13, paras_fixed)
                file_.write(str_optim.format(*line))

                val = dict_['EDUCATION']['coeffs'][2]
                line = format_opt_parameters(val, 14, paras_fixed)
                file_.write(str_optim.format(*line))

                file_.write('\n')
                str_ = '{0:<10} {1:>15} \n'
                file_.write(str_.format('start', dict_[flag]['start']))
                file_.write(str_.format('max', dict_[flag]['max']))

                file_.write('\n')

            if flag in ['OCCUPATION A', 'OCCUPATION B']:
                identifier = None
                if flag == 'OCCUPATION A':
                    identifier = 0
                if flag == 'OCCUPATION B':
                    identifier = 6

                file_.write(flag + '\n\n')

                # Coefficient
                for j in range(6):
                    val = dict_[flag]['coeffs'][j]
                    line = format_opt_parameters(val, identifier, paras_fixed)
                    identifier += 1

                    file_.write(str_optim.format(*line))

                file_.write('\n')

            if flag in ['SCIPY-POWELL', 'SCIPY-BFGS']:

                # This function can also be used to print out initialization
                # files without any optimization options. This is enough for
                # simulation tasks.
                if flag not in dict_.keys():
                    continue

                file_.write(flag.upper() + '\n\n')

                for keys_ in dict_[flag]:

                    if keys_ in ['maxfun']:
                        str_ = '{0:<10} {1:>20} \n'
                        file_.write(str_.format(keys_, dict_[flag][keys_]))
                    else:
                        str_ = '{0:<10} {1:20.4f} \n'
                        file_.write(str_.format(keys_, dict_[flag][keys_]))

                file_.write('\n')


def format_opt_parameters(val, identifier, paras_fixed):
    """ This function formats the values depending on whether they are fixed
    during the optimization or not.
    """
    # Initialize baseline line
    line = ['coeff', val, ' ']
    if paras_fixed[identifier]:
        line[-1] = '!'

    # Finishing
    return line