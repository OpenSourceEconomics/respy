import os
import shlex

import numpy as np

from respy.python.shared.shared_constants import INADMISSIBILITY_PENALTY
from respy.python.record.record_warning import record_warning
from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.shared.shared_constants import OPT_EST_FORT
from respy.python.shared.shared_constants import OPT_EST_PYTH
from respy.python.shared.shared_constants import OPT_AMB_FORT
from respy.python.shared.shared_constants import OPT_AMB_PYTH
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.python.shared.shared_constants import TINY_FLOAT

OPTIMIZERS = OPT_EST_FORT + OPT_EST_PYTH + OPT_AMB_FORT + OPT_AMB_PYTH


def get_log_likl(contribs):
    """ Aggregate contributions to the likelihood value.
    """
    # We want to make sure to note if the we truncated zero-probability agents.
    if sum(np.abs(contribs) > HUGE_FLOAT) > 0:
        record_warning(5)

    crit_val = -np.mean(np.clip(np.log(contribs), -HUGE_FLOAT, HUGE_FLOAT))

    return crit_val


def check_optimization_parameters(x):
    """ Check optimization parameters.
    """
    # Perform checks
    assert (isinstance(x, np.ndarray))
    assert (x.dtype == np.float)
    assert (np.all(np.isfinite(x)))

    # Finishing
    return True


def dist_optim_paras(x_all_curre, is_debug, info=None):
    """ Update parameter values. The np.array type is maintained.
    """
    # Checks
    if is_debug:
        check_optimization_parameters(x_all_curre)

    # Level of Ambiguity
    level = x_all_curre[0:1] ** 2

    # Occupation A
    coeffs_a = x_all_curre[1:7]

    # Occupation B
    coeffs_b = x_all_curre[7:13]

    # Education
    coeffs_edu = x_all_curre[13:16]

    # Home
    coeffs_home = x_all_curre[16:17]

    # Cholesky
    shocks_cholesky, info = get_cholesky(x_all_curre, info)

    # Checks
    if is_debug:
        args = (level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
                shocks_cholesky)
        assert check_model_parameters(*args)

    # Collect arguments
    args = (level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky)

    # Finishing
    return args


def get_cholesky(x, info=None):
    """ Construct the Cholesky matrix.
    """
    shocks_cholesky = np.tile(0.0, (4, 4))
    shocks_cholesky[0, :1] = x[17:18]
    shocks_cholesky[1, :2] = x[18:20]
    shocks_cholesky[2, :3] = x[20:23]
    shocks_cholesky[3, :4] = x[23:27]

    # Stabilization
    if info is not None:
        info = 0

    # We need to ensure that the diagonal elements are larger than zero
    # during an estimation. However, we want to allow for the special case of
    # total absence of randomness for testing purposes of simulated datasets.
    if not (np.count_nonzero(shocks_cholesky) == 0):
        shocks_cov = np.matmul(shocks_cholesky, shocks_cholesky.T)
        for i in range(4):
            if np.abs(shocks_cov[i, i]) < TINY_FLOAT:
                shocks_cholesky[i, i] = np.sqrt(TINY_FLOAT)
                if info is not None:
                    info = 1

    if info is not None:
        return shocks_cholesky, info
    else:
        return shocks_cholesky, None


def get_total_values(period, num_periods, delta, rewards_systematic, draws,
        edu_max, edu_start, mapping_state_idx, periods_emax, k, states_all):
    """ Get total value of all possible states.
    """
    # Initialize containers
    rewards_ex_post = np.tile(np.nan, 4)

    # Calculate ex post rewards
    for j in [0, 1]:
        rewards_ex_post[j] = rewards_systematic[j] * draws[j]

    for j in [2, 3]:
        rewards_ex_post[j] = rewards_systematic[j] + draws[j]

    # Get future values
    if period != (num_periods - 1):
        emaxs, is_inadmissible = get_emaxs(edu_max, edu_start, mapping_state_idx, period, periods_emax, k,
                                                   states_all)
    else:
        is_inadmissible = False
        emaxs = np.tile(0.0, 4)

    # Calculate total utilities
    total_values = rewards_ex_post + delta * emaxs

    # This is required to ensure that the agent does not choose any
    # inadmissible states. If the state is inadmissible emaxs takes
    # value zero. This aligns the treatment of inadmissible values with the
    # original paper.
    if is_inadmissible:
        total_values[2] += INADMISSIBILITY_PENALTY

    # Finishing
    return total_values


def get_emaxs(edu_max, edu_start, mapping_state_idx, period,
             periods_emax, k, states_all):
    """ Get emaxs for additional choices.
    """
    # Distribute state space
    exp_a, exp_b, edu, _ = states_all[period, k, :]

    # Future utilities
    emaxs = np.tile(np.nan, 4)

    # Working in Occupation A
    future_idx = mapping_state_idx[period + 1, exp_a + 1, exp_b, edu, 0]
    emaxs[0] = periods_emax[period + 1, future_idx]

    # Working in Occupation B
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b + 1, edu, 0]
    emaxs[1] = periods_emax[period + 1, future_idx]

    # Increasing schooling. Note that adding an additional year
    # of schooling is only possible for those that have strictly
    # less than the maximum level of additional education allowed.
    is_inadmissible = (edu >= edu_max - edu_start)
    if is_inadmissible:
        emaxs[2] = 0.00
    else:
        future_idx = mapping_state_idx[period + 1, exp_a, exp_b, edu + 1, 1]
        emaxs[2] = periods_emax[period + 1, future_idx]

    # Staying at home
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b, edu, 0]
    emaxs[3] = periods_emax[period + 1, future_idx]

    # Finishing
    return emaxs, is_inadmissible


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


def cholesky_to_coeffs(shocks_cholesky):
    """ This function maps the Cholesky factor into the coefficients as
    specified in the initialization file.
    """

    shocks_cov = np.matmul(shocks_cholesky, shocks_cholesky.T)
    for i in range(4):
        shocks_cov[i, i] = np.sqrt(shocks_cov[i, i])

    shocks_coeffs = shocks_cov[np.triu_indices(4)].tolist()

    return shocks_coeffs


def add_solution(respy_obj, periods_rewards_systematic,
        states_number_period, mapping_state_idx, periods_emax, states_all):
    """ Add solution to class instance.
    """
    respy_obj.unlock()

    respy_obj.set_attr('periods_rewards_systematic', periods_rewards_systematic)

    respy_obj.set_attr('states_number_period', states_number_period)

    respy_obj.set_attr('mapping_state_idx', mapping_state_idx)

    respy_obj.set_attr('periods_emax', periods_emax)

    respy_obj.set_attr('states_all', states_all)

    respy_obj.set_attr('is_solved', True)

    respy_obj.lock()

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
        # Agent index
        if i == 1:
            assert (data_frame.max(axis=0)[i] == (max_ - 1))

        if (i == 0) and (which == 'sim'):
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
        assert (data_frame[0].value_counts().iloc[agent] == num_periods)

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
    level = model_paras['level']
    coeffs_a = model_paras['coeffs_a']
    coeffs_b = model_paras['coeffs_b']

    coeffs_edu = model_paras['coeffs_edu']
    coeffs_home = model_paras['coeffs_home']
    shocks_cholesky = model_paras['shocks_cholesky']

    if is_debug:
        args = (level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
                shocks_cholesky)
        assert (check_model_parameters(*args))

    # Finishing
    return level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky


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


def check_model_parameters(level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cholesky):
    """ Check the integrity of all model parameters.
    """
    # Checks for all arguments
    args = (level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky)
    for coeffs in args:
        assert (isinstance(coeffs, np.ndarray))
        assert (np.all(np.isfinite(coeffs)))
        assert (coeffs.dtype == 'float')

    # Check for level of ambiguity
    assert (level >= 0)

    # Checks for occupations
    assert (coeffs_a.size == 6)
    assert (coeffs_b.size == 6)
    assert (coeffs_edu.size == 3)
    assert (coeffs_home.size == 1)

    # Checks shock matrix
    assert (shocks_cholesky.shape == (4, 4))
    np.allclose(shocks_cholesky, np.tril(shocks_cholesky))

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

    paras_fixed = dict_['AMBIGUITY']['fixed'][:]
    paras_fixed += dict_['OCCUPATION A']['fixed'][:]
    paras_fixed += dict_['OCCUPATION B']['fixed'][:]
    paras_fixed += dict_['EDUCATION']['fixed'][:]
    paras_fixed += dict_['HOME']['fixed'][:]

    # TODO: This needs to be better understood before a release.
    try:
        paras_fixed += dict_['SHOCKS']['fixed'][:]
    except ValueError:
        paras_fixed += dict_['SHOCKS']['fixed'][:].tolist()

    str_optim = '{0:<10} {1:25.15f} {2:>5}\n'

    # Construct labels. This ensures that the initialization files always look
    # identical.
    labels = ['BASICS', 'AMBIGUITY', 'OCCUPATION A', 'OCCUPATION B']
    labels += ['EDUCATION', 'HOME', 'SHOCKS', 'SOLUTION']
    labels += ['SIMULATION', 'ESTIMATION', 'DERIVATIVES', 'SCALING']
    labels += ['PROGRAM', 'INTERPOLATION']
    labels += OPT_EST_FORT + OPT_EST_PYTH + ['SCIPY-SLSQP', 'FORT-SLSQP']

    # Create initialization.
    with open(file_name, 'w') as file_:

        for flag in labels:
            if flag in ['BASICS']:

                file_.write('BASICS\n\n')

                str_ = '{0:<10} {1:>20}\n'
                file_.write(str_.format('periods', dict_[flag]['periods']))

                str_ = '{0:<10} {1:20.4f}\n'
                file_.write(str_.format('delta', dict_[flag]['delta']))

                file_.write('\n')

            if flag in ['HOME']:

                file_.write(flag.upper() + '\n\n')

                val = dict_['HOME']['coeffs'][0]
                line = format_opt_parameters(val, 16, paras_fixed)
                file_.write(str_optim.format(*line))

                file_.write('\n')

            if flag in ['SOLUTION', 'SIMULATION', 'PROGRAM', 'INTERPOLATION',
                        'ESTIMATION', 'SCALING', 'DERIVATIVES']:

                file_.write(flag.upper() + '\n\n')
                keys = list(dict_[flag].keys())
                keys.sort()
                for key_ in keys:

                    if key_ in ['tau']:
                        str_ = '{0:<10} {1:20.15f}\n'
                        file_.write(str_.format(key_, dict_[flag][key_]))
                    else:
                        str_ = '{0:<10} {1:>20}\n'
                        file_.write(str_.format(key_, str(dict_[flag][key_])))

                file_.write('\n')

            if flag in ['SHOCKS']:

                # Type conversion
                file_.write(flag.upper() + '\n\n')

                for i in range(10):
                    val = dict_['SHOCKS']['coeffs'][i]
                    line = format_opt_parameters(val, 17 + i, paras_fixed)
                    file_.write(str_optim.format(*line))
                file_.write('\n')

            if flag in ['EDUCATION']:

                file_.write(flag.upper() + '\n\n')

                val = dict_['EDUCATION']['coeffs'][0]
                line = format_opt_parameters(val, 13, paras_fixed)
                file_.write(str_optim.format(*line))

                val = dict_['EDUCATION']['coeffs'][1]
                line = format_opt_parameters(val, 14, paras_fixed)
                file_.write(str_optim.format(*line))

                val = dict_['EDUCATION']['coeffs'][2]
                line = format_opt_parameters(val, 15, paras_fixed)
                file_.write(str_optim.format(*line))

                file_.write('\n')
                str_ = '{0:<10} {1:>20}\n'
                file_.write(str_.format('start', dict_[flag]['start']))
                file_.write(str_.format('max', dict_[flag]['max']))

                file_.write('\n')

            if flag in ['AMBIGUITY']:
                file_.write(flag.upper() + '\n\n')

                val = dict_['AMBIGUITY']['coeffs'][0]
                line = format_opt_parameters(val, 0, paras_fixed)
                file_.write(str_optim.format(*line))

                str_ = '{0:<10} {1:>20}\n'
                file_.write(str_.format('measure', dict_[flag]['measure']))

                file_.write('\n')

            if flag in ['OCCUPATION A', 'OCCUPATION B']:
                identifier = None
                if flag == 'OCCUPATION A':
                    identifier = 1
                if flag == 'OCCUPATION B':
                    identifier = 7

                file_.write(flag + '\n\n')

                # Coefficient
                for j in range(6):
                    val = dict_[flag]['coeffs'][j]
                    line = format_opt_parameters(val, identifier, paras_fixed)
                    identifier += 1

                    file_.write(str_optim.format(*line))

                file_.write('\n')

            if flag in OPTIMIZERS:

                # This function can also be used to print out initialization
                # files without any optimization options. This is enough for
                # simulation tasks.
                if flag not in dict_.keys():
                    continue

                file_.write(flag.upper() + '\n\n')
                keys = list(dict_[flag].keys())
                keys.sort()
                for key_ in keys:

                    # The derivative is approximated with the dfunc_eps
                    # attribute. All derivatives in the package have a single
                    # approximation step size.
                    if key_ == 'eps':
                        continue

                    if key_ in ['maxfun', 'npt', 'maxiter']:
                        str_ = '{0:<10} {1:>20}\n'
                        file_.write(str_.format(key_, dict_[flag][key_]))
                    else:
                        str_ = '{0:<10} {1:20.15f}\n'
                        file_.write(str_.format(key_, dict_[flag][key_]))

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


def process_est_log():
    """ Process information in the est.respy.log for further inspection.
    """
    rslt = dict()

    rslt['value_final'] = None
    rslt['paras_final'] = []

    is_report = False
    is_final = False
    with open('est.respy.log') as in_file:
        for _, line in enumerate(in_file.readlines()):
            list_ = shlex.split(line)

            if not list_:
                continue

            if list_[0] == 'ESTIMATION':
                is_report = True
            try:
                if list_[1] == 'Final':
                    is_final = True
            except IndexError:
                pass

            # Collect the final value of criterion function
            if is_report and (list_[0] == 'Criterion'):
                rslt['value_final'] = float(list_[1])

            if is_final:
                try:
                    rslt['paras_final'] += [float(list_[1])]
                except ValueError:
                    pass

    # Type transformations
    rslt['paras_final'] = np.array(rslt['paras_final'])

    return rslt


def get_est_info():
    """ This function reads in the parameters from the last step of a
    previous estimation run.
    """
    def _process_value(input_):
        try:
            value = float(input_)
        except ValueError:
            value = '---'

        return value

    rslt = dict()

    paras_start, paras_step, paras_current = [], [], []
    with open('est.respy.info') as in_file:
        for i, line in enumerate(in_file.readlines()):

            list_ = shlex.split(line)

            if i == 5:
                value_start = _process_value(list_[0])
                value_step = _process_value(list_[1])
                value_current = _process_value(list_[2])

            if i in range(12, 39):
                paras_start += [float(list_[1])]
                paras_step += [float(list_[2])]
                paras_current += [float(list_[3])]

            if i == 67:
                num_step = int(float(list_[3]))

            if i == 69:
                num_eval = int(float(list_[3]))

    rslt['paras_current'] = np.array(paras_current)
    rslt['paras_start'] = np.array(paras_start)
    rslt['paras_step'] = np.array(paras_step)

    rslt['value_current'] = value_current
    rslt['value_start'] = value_start
    rslt['value_step'] = value_step

    rslt['num_eval'] = num_eval
    rslt['num_step'] = num_step

    # Finishing
    return rslt
