import numpy as np

import linecache
import atexit
import shlex
import glob
import os

from respy.python.shared.shared_constants import INADMISSIBILITY_PENALTY
from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.record.record_warning import record_warning
from respy.python.shared.shared_constants import OPT_AMB_FORT
from respy.python.shared.shared_constants import OPT_AMB_PYTH
from respy.python.shared.shared_constants import OPT_EST_FORT
from respy.python.shared.shared_constants import OPT_EST_PYTH
from respy.python.shared.shared_constants import PRINT_FLOAT
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.python.shared.shared_constants import TINY_FLOAT
from respy.python.shared.shared_constants import NUM_PARAS
from respy.custom_exceptions import MaxfunError
from respy.custom_exceptions import UserError

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


def dist_econ_paras(x_all_curre):
    """ Update parameter values. The np.array type is maintained.
    """
    # Discount rates
    delta = x_all_curre[0:1]

    # Level of Ambiguity
    level = x_all_curre[1:2]

    # Occupation A
    coeffs_a = x_all_curre[2:11]

    # Occupation B
    coeffs_b = x_all_curre[11:20]

    # Education
    coeffs_edu = x_all_curre[20:24]

    # Home
    coeffs_home = x_all_curre[24:25]

    shocks_coeffs = x_all_curre[25:NUM_PARAS]
    for i in [0, 4, 7, 9]:
        shocks_coeffs[i] **= 2

    shocks = np.zeros((4, 4))
    shocks[0, :] = shocks_coeffs[0:4]
    shocks[1, 1:] = shocks_coeffs[4:7]
    shocks[2, 2:] = shocks_coeffs[7:9]
    shocks[3, 3:] = shocks_coeffs[9:10]

    shocks_cov = shocks + shocks.T - np.diag(shocks.diagonal())

    # Collect arguments
    args = (delta, level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cov)

    # Finishing
    return args


def dist_optim_paras(x_all_curre, is_debug, info=None):
    """ Update parameter values. The np.array type is maintained.
    """
    # Checks
    if is_debug:
        check_optimization_parameters(x_all_curre)

    optim_paras = dict()

    # Discount rate
    optim_paras['delta'] = max(x_all_curre[0:1], 0.00)

    # Level of Ambiguity
    optim_paras['level'] = max(x_all_curre[1:2], 0.00)

    # Occupation A
    optim_paras['coeffs_a'] = x_all_curre[2:11]

    # Occupation B
    optim_paras['coeffs_b'] = x_all_curre[11:20]

    # Education
    optim_paras['coeffs_edu'] = x_all_curre[20:24]

    # Home
    optim_paras['coeffs_home'] = x_all_curre[24:25]

    # Cholesky
    optim_paras['shocks_cholesky'], info = extract_cholesky(x_all_curre, info)

    # Checks
    if is_debug:
        assert check_model_parameters(optim_paras)

    # Finishing
    return optim_paras


def extract_cholesky(x, info=None):
    """ Construct the Cholesky matrix.
    """
    shocks_cholesky = np.tile(0.0, (4, 4))
    shocks_cholesky[0, :1] = x[25:26]
    shocks_cholesky[1, :2] = x[26:28]
    shocks_cholesky[2, :3] = x[28:31]
    shocks_cholesky[3, :4] = x[31:NUM_PARAS]

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


def get_total_values(period, num_periods, optim_paras, rewards_systematic,
        draws, edu_max, edu_start, mapping_state_idx, periods_emax, k,
        states_all):
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
    total_values = rewards_ex_post + optim_paras['delta'] * emaxs

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
    exp_a, exp_b, edu, _, type_ = states_all[period, k, :]

    # Future utilities
    emaxs = np.tile(np.nan, 4)

    # Working in Occupation A
    future_idx = mapping_state_idx[period + 1, exp_a + 1, exp_b, edu, 0, type_]
    emaxs[0] = periods_emax[period + 1, future_idx]

    # Working in Occupation B
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b + 1, edu, 0, type_]
    emaxs[1] = periods_emax[period + 1, future_idx]

    # Increasing schooling. Note that adding an additional year
    # of schooling is only possible for those that have strictly
    # less than the maximum level of additional education allowed.
    is_inadmissible = (edu >= edu_max - edu_start)
    if is_inadmissible:
        emaxs[2] = 0.00
    else:
        future_idx = mapping_state_idx[period + 1, exp_a, exp_b, edu + 1, 1, type_]
        emaxs[2] = periods_emax[period + 1, future_idx]

    # Staying at home
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b, edu, 0, type_]
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


def check_model_parameters(optim_paras):
    """ Check the integrity of all model parameters.
    """
    # Checks for all arguments
    keys = []
    keys += ['coeffs_a', 'coeffs_b', 'coeffs_edu', 'coeffs_home']
    keys += ['level', 'shocks_cholesky', 'delta']

    for key in keys:
        assert (isinstance(optim_paras[key], np.ndarray))
        assert (np.all(np.isfinite(optim_paras[key])))
        assert (optim_paras[key].dtype == 'float')
        assert (np.all(abs(optim_paras[key]) < PRINT_FLOAT))

    # Check for discount rate
    assert (optim_paras['delta'] >= 0)

    # Check for level of ambiguity
    assert (optim_paras['level'] >= 0)

    # Checks for occupations
    assert (optim_paras['coeffs_a'].size == 9)
    assert (optim_paras['coeffs_b'].size == 9)
    assert (optim_paras['coeffs_edu'].size == 4)
    assert (optim_paras['coeffs_home'].size == 1)

    # Checks shock matrix
    assert (optim_paras['shocks_cholesky'].shape == (4, 4))
    np.allclose(optim_paras['shocks_cholesky'],
        np.tril(optim_paras['shocks_cholesky']))

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


def transform_disturbances(draws, shocks_mean, shocks_cholesky):
    """ Transform the standard normal deviates to the relevant distribution.

    """
    # Transfer draws to relevant distribution
    draws_transformed = draws.copy()
    draws_transformed = np.dot(shocks_cholesky, draws_transformed.T).T

    for j in range(4):
        draws_transformed[:, j] = draws_transformed[:, j] + shocks_mean[j]

    for j in range(2):
        draws_transformed[:, j] = \
            np.clip(np.exp(draws_transformed[:, j]), 0.0, HUGE_FLOAT)

    # Finishing
    return draws_transformed


def generate_optimizer_options(which, optim_paras):

    dict_ = dict()

    if which == 'SCIPY-BFGS':
        dict_['gtol'] = np.random.uniform(0.0000001, 0.1)
        dict_['maxiter'] = np.random.randint(1, 10)
        dict_['eps'] = np.random.uniform(1e-9, 1e-6)

    elif which == 'SCIPY-LBFGSB':
        dict_['factr'] = np.random.uniform(10, 100)
        dict_['pgtol'] = np.random.uniform(1e-6, 1e-4)
        dict_['maxiter'] = np.random.randint(1, 10)
        dict_['maxls'] = np.random.randint(1, 10)
        dict_['m'] = np.random.randint(1, 10)
        dict_['eps'] = np.random.uniform(1e-9, 1e-6)

    elif which == 'SCIPY-POWELL':
        dict_['xtol'] = np.random.uniform(0.0000001, 0.1)
        dict_['ftol'] = np.random.uniform(0.0000001, 0.1)
        dict_['maxfun'] = np.random.randint(1, 100)
        dict_['maxiter'] = np.random.randint(1, 100)

    elif which in ['FORT-NEWUOA', 'FORT-BOBYQA']:
        rhobeg = np.random.uniform(0.0000001, 0.001)
        dict_['maxfun'] = np.random.randint(1, 100)
        dict_['rhobeg'] = rhobeg
        dict_['rhoend'] = np.random.uniform(0.01, 0.99) * rhobeg

        # It is not recommended that N is larger than upper as the code might
        # break down due to a segmentation fault. See the source files for the
        # absolute upper bounds.
        assert sum(optim_paras['paras_fixed']) != NUM_PARAS
        lower = (NUM_PARAS - sum(optim_paras['paras_fixed'])) + 2
        upper = (2 * (NUM_PARAS - sum(optim_paras['paras_fixed'])) + 1)
        dict_['npt'] = np.random.randint(lower, upper + 1)

    elif which == 'FORT-BFGS':
        dict_['maxiter'] = np.random.randint(1, 100)
        dict_['stpmx'] = np.random.uniform(75, 125)
        dict_['gtol'] = np.random.uniform(0.0001, 0.1)
        dict_['eps'] = np.random.uniform(1e-9, 1e-6)

    elif which in ['FORT-SLSQP', 'SCIPY-SLSQP']:
        dict_['maxiter'] = np.random.randint(50, 100)
        dict_['ftol'] = np.random.uniform(1e-9, 1e-6)
        dict_['eps'] = np.random.uniform(1e-9, 1e-6)

    return dict_


def print_init_dict(dict_, file_name='test.respy.ini'):
    """ Print initialization dictionary to file. The different formatting
    makes the file rather involved. The resulting initialization files are
    read by PYTHON and FORTRAN routines. Thus, the formatting with respect to
    the number of decimal places is rather small.
    """
    # Antibugging.
    assert (isinstance(dict_, dict))

    paras_fixed = dict_['BASICS']['fixed'][:]
    paras_fixed += dict_['AMBIGUITY']['fixed'][:]
    paras_fixed += dict_['OCCUPATION A']['fixed'][:]
    paras_fixed += dict_['OCCUPATION B']['fixed'][:]
    paras_fixed += dict_['EDUCATION']['fixed'][:]
    paras_fixed += dict_['HOME']['fixed'][:]
    paras_fixed += dict_['SHOCKS']['fixed'][:]

    paras_bounds = dict_['BASICS']['bounds'][:]
    paras_bounds += dict_['AMBIGUITY']['bounds'][:]
    paras_bounds += dict_['OCCUPATION A']['bounds'][:]
    paras_bounds += dict_['OCCUPATION B']['bounds'][:]
    paras_bounds += dict_['EDUCATION']['bounds'][:]
    paras_bounds += dict_['HOME']['bounds'][:]
    paras_bounds += dict_['SHOCKS']['bounds'][:]

    str_optim = '{0:<10} {1:25.15f} {2:>5} {3:>15}\n'
    str_types = '{0:<10} {1:25.15f}\n'

    # Construct labels. This ensures that the initialization files always look
    # identical.
    labels = ['BASICS', 'AMBIGUITY', 'OCCUPATION A', 'OCCUPATION B']
    labels += ['EDUCATION', 'HOME', 'SHOCKS', 'TYPES', 'SOLUTION']
    labels += ['SIMULATION', 'ESTIMATION', 'DERIVATIVES', 'PRECONDITIONING']
    labels += ['PROGRAM', 'INTERPOLATION']
    labels += OPT_EST_FORT + OPT_EST_PYTH + ['SCIPY-SLSQP', 'FORT-SLSQP']

    # Create initialization.
    with open(file_name, 'w') as file_:

        for flag in labels:
            if flag in ['BASICS']:

                file_.write('BASICS\n\n')

                str_ = '{0:<10} {1:>20}\n'
                file_.write(str_.format('periods', dict_[flag]['periods']))

                val = dict_['BASICS']['coeffs'][0]
                line = format_opt_parameters(val, 0, paras_fixed, paras_bounds)
                file_.write(str_optim.format(*line))

                file_.write('\n')

            if flag in ['TYPES']:
                num_types = len(dict_['TYPES']['shares'])
                file_.write(flag.upper() + '\n\n')

                for i in range(num_types):
                    file_.write(str_types.format('share', dict_[flag]['shares'][i]))
                    # The first group serves as the baseline and thus there
                    # are no shifts printed to the file.
                    if i == 0:
                        pass
                    else:
                        for j in range(4):
                            line = str_types.format('shift', dict_[flag][
                                'shifts'][i, j])
                            file_.write(line)

                    file_.write('\n')

            if flag in ['HOME']:

                file_.write(flag.upper() + '\n\n')

                val = dict_['HOME']['coeffs'][0]
                line = format_opt_parameters(val, 24, paras_fixed, paras_bounds)
                file_.write(str_optim.format(*line))

                file_.write('\n')

            if flag in ['SOLUTION', 'SIMULATION', 'PROGRAM', 'INTERPOLATION',
                        'ESTIMATION', 'PRECONDITIONING', 'DERIVATIVES']:

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
                    line = format_opt_parameters(val, 25 + i, paras_fixed,
                        paras_bounds)
                    file_.write(str_optim.format(*line))
                file_.write('\n')

            if flag in ['EDUCATION']:

                file_.write(flag.upper() + '\n\n')

                for i in range(4):
                    val = dict_['EDUCATION']['coeffs'][i]
                    line = format_opt_parameters(val, i + 20, paras_fixed,
                        paras_bounds)
                    file_.write(str_optim.format(*line))

                file_.write('\n')
                str_ = '{0:<10} {1:>20}\n'
                file_.write(str_.format('start', dict_[flag]['start']))
                file_.write(str_.format('max', dict_[flag]['max']))

                file_.write('\n')

            if flag in ['AMBIGUITY']:
                file_.write(flag.upper() + '\n\n')

                val = dict_['AMBIGUITY']['coeffs'][0]
                line = format_opt_parameters(val, 1, paras_fixed, paras_bounds)
                file_.write(str_optim.format(*line))

                str_ = '{0:<10} {1:>20}\n'
                file_.write(str_.format('measure', dict_[flag]['measure']))
                file_.write(str_.format('mean', str(dict_[flag]['mean'])))

                file_.write('\n')

            if flag in ['OCCUPATION A', 'OCCUPATION B']:
                identifier = None
                if flag == 'OCCUPATION A':
                    identifier = 2
                if flag == 'OCCUPATION B':
                    identifier = 11

                file_.write(flag + '\n\n')

                # Coefficient
                for j in range(9):
                    val = dict_[flag]['coeffs'][j]
                    line = format_opt_parameters(val, identifier, paras_fixed,
                                                 paras_bounds)
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

                    if key_ in ['maxfun', 'npt', 'maxiter', 'm', 'maxls']:
                        str_ = '{0:<10} {1:>20}\n'
                        file_.write(str_.format(key_, dict_[flag][key_]))
                    else:
                        str_ = '{0:<10} {1:20.15f}\n'
                        file_.write(str_.format(key_, dict_[flag][key_]))

                file_.write('\n')


def format_opt_parameters(val, identifier, paras_fixed, paras_bounds):
    """ This function formats the values depending on whether they are fixed
    during the optimization or not.
    """
    # Initialize baseline line
    line = ['coeff', val, ' ', ' ']
    if paras_fixed[identifier]:
        line[-2] = '!'

    # Check if any bounds defined
    bounds = paras_bounds[identifier]
    if any(x is not None for x in bounds):
        line[-1] = '(' + str(bounds[0]) + ',' + str(bounds[1]) + ')'

    # Finishing
    return line


def apply_scaling(x, precond_matrix, request):
    """ Apply or revert the preconditioning step
    """
    if request == 'do':
        out = np.dot(precond_matrix, x)
    elif request == 'undo':
        out = np.dot(np.linalg.pinv(precond_matrix), x)
    else:
        raise AssertionError

    return out


def get_est_info():
    """ This function reads in the parameters from the last step of a
    previous estimation run.
    """
    def _process_value(input_, type_):
        try:
            if type_ == 'float':
                value = float(input_)
            elif type_ == 'int':
                value = int(input_)
        except ValueError:
            value = '---'

        return value

    # We need to make sure that the updating file actually exists.
    if not os.path.exists("est.respy.info"):
        msg = 'Parameter update impossible as '
        msg += 'file est.respy.info does not exist'
        raise UserError(msg)

    # Initialize container and ensure a fresh start processing the file
    linecache.clearcache()
    rslt = dict()

    # Value of the criterion function
    line = shlex.split(linecache.getline('est.respy.info', 6))
    for key_ in ['start', 'step', 'current']:
        rslt['value_' + key_] = _process_value(line.pop(0), 'float')

    # Total number of evaluations and steps
    line = shlex.split(linecache.getline('est.respy.info', 42))
    rslt['num_step'] = _process_value(line[3], 'int')

    line = shlex.split(linecache.getline('est.respy.info', 44))
    rslt['num_eval'] = _process_value(line[3], 'int')

    # Parameter values
    for i, key_ in enumerate(['start', 'step', 'current']):
        rslt['paras_' + key_] = []
        for j in range(13, NUM_PARAS + 13):
            line = shlex.split(linecache.getline('est.respy.info', j))
            rslt['paras_' + key_] += [_process_value(line[i + 1], 'float')]
        rslt['paras_' + key_] = np.array(rslt['paras_' + key_])

    # Finishing
    return rslt


@atexit.register
def remove_scratch_files():
    """ This function removes all scratch files.
    """
    for fname in glob.glob('.*.respy.scratch'):
        os.unlink(fname)


def get_optim_paras(optim_paras, which, is_debug):
    """ Get optimization parameters.
    """
    # Checks
    if is_debug:
        assert check_model_parameters(optim_paras)

    # Initialize container
    x = np.tile(np.nan, NUM_PARAS)

    # Discount rate
    x[0:1] = optim_paras['delta']

    # Level of Ambiguity
    x[1:2] = optim_paras['level']

    # Occupation A
    x[2:11] = optim_paras['coeffs_a']

    # Occupation B
    x[11:20] = optim_paras['coeffs_b']

    # Education
    x[20:24] = optim_paras['coeffs_edu']

    # Home
    x[24:25] = optim_paras['coeffs_home']

    # Shocks
    x[25:NUM_PARAS] = optim_paras['shocks_cholesky'][np.tril_indices(4)]

    # Checks
    if is_debug:
        check_optimization_parameters(x)

    # Select subset
    if which == 'free':
        x_free_curre = []
        for i in range(NUM_PARAS):
            if not optim_paras['paras_fixed'][i]:
                x_free_curre += [x[i]]

        x = np.array(x_free_curre)

    # Finishing
    return x


def covariance_to_correlation(cov):
    """ This function constructs the correlation matrix from the information on
    the covariances.
    """
    # Auxiliary objects
    corr = np.tile(np.nan, cov.shape)
    nrows = cov.shape[0]

    # This special case is maintained for testing purposes.
    is_deterministic = (np.count_nonzero(cov) == 0)
    if is_deterministic:
        return np.zeros((nrows, nrows))

    for i in range(nrows):
        for j in range(nrows):
            corr[i, j] = cov[i, j] / (np.sqrt(cov[i, i]) * np.sqrt(cov[j, j]))

    return corr


def correlation_to_covariance(corr, sd):
    """ This function constructs the covariance matrix from the information on
    the correlations.
    """
    # Auxiliary objects
    cov = np.tile(np.nan, corr.shape)
    nrows = corr.shape[0]

    # This special case is maintained for testing purposes.
    is_deterministic = (np.count_nonzero(sd) == 0)
    if is_deterministic:
        return np.zeros((nrows, nrows))

    for i in range(nrows):
        for j in range(nrows):
            cov[i, j] = corr[i, j] * sd[j] * sd[i]

    return cov


def check_early_termination(maxfun, num_eval):
    """ This function checks for reasons that require an early termination of
    the optimization procedure.
    """
    # We want an early termination if the number of function evaluations is
    # already at the maximum number requested. This is not strictly enforced
    # in some of the SCIPY algorithms.
    if maxfun == num_eval:
        raise MaxfunError

    # We also want the opportunity for an immediate, but gentle termination from
    # the user.
    if os.path.exists('.stop.respy.scratch'):
        raise MaxfunError



