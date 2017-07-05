import numpy as np

import linecache
import shlex
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
    # Auxiliary objects
    num_paras = len(x_all_curre)
    num_types = int(len(x_all_curre[50:]) / 6) + 1

    # Discount rates
    delta = x_all_curre[0:1]

    # Level of Ambiguity
    level = x_all_curre[1:2]

    # Common Returns
    coeffs_common = x_all_curre[2:4]

    # Occupation A
    coeffs_a = x_all_curre[4:17]

    # Occupation B
    coeffs_b = x_all_curre[17:30]

    # Education
    coeffs_edu = x_all_curre[30:37]

    # Home
    coeffs_home = x_all_curre[37:40]

    shocks_coeffs = x_all_curre[40:50]
    for i in [0, 4, 7, 9]:
        shocks_coeffs[i] **= 2

    shocks = np.zeros((4, 4))
    shocks[0, :] = shocks_coeffs[0:4]
    shocks[1, 1:] = shocks_coeffs[4:7]
    shocks[2, 2:] = shocks_coeffs[7:9]
    shocks[3, 3:] = shocks_coeffs[9:10]

    shocks_cov = shocks + shocks.T - np.diag(shocks.diagonal())

    # Type Shares
    type_shares = x_all_curre[50:50 + (num_types - 1) * 2]
    type_shares = np.concatenate((np.tile(0.0, 2), type_shares), axis=0)
    
    type_shifts = np.reshape(x_all_curre[50 + (num_types - 1) * 2:num_paras], (num_types - 1, 4))
    type_shifts = np.concatenate((np.tile(0.0, (1, 4)), type_shifts), axis=0)

    # Collect arguments
    args = (delta, level, coeffs_common, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            type_shares, type_shifts)

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

    # Common Rewards
    optim_paras['coeffs_common'] = x_all_curre[2:4]

    # Occupation A
    optim_paras['coeffs_a'] = x_all_curre[4:17]

    # Occupation B
    optim_paras['coeffs_b'] = x_all_curre[17:30]

    # Education
    optim_paras['coeffs_edu'] = x_all_curre[30:37]

    # Home
    optim_paras['coeffs_home'] = x_all_curre[37:40]

    # Cholesky
    optim_paras['shocks_cholesky'], info = extract_cholesky(x_all_curre, info)

    type_shares, type_shifts = extract_type_information(x_all_curre)
    optim_paras['type_shares'] = type_shares
    optim_paras['type_shifts'] = type_shifts

    # Checks
    if is_debug:
        assert check_model_parameters(optim_paras)

    # Finishing
    return optim_paras


def get_conditional_probabilities(type_shares, edu_start):
    """ This function calculates the conditional choice probabilities based on the mulitnomial
    logit model for one particular initial condition.
    """
    # Auxiliary objects
    num_types = int(len(type_shares) / 2)
    probs = np.tile(np.nan, num_types)
    for i in range(num_types):
        lower, upper = i * 2, (i + 1) * 2
        probs[i] = np.exp(np.sum(type_shares[lower:upper] * [1.0, edu_start]))

    # Scaling
    probs = probs / sum(probs)

    return probs


def extract_type_information(x):
    """ This function extracts the information about types from the estimation vector.
    """

    num_types = int(len(x[50:]) / 6) + 1

    # Type shares
    type_shares = x[50:50 + (num_types - 1) * 2]
    type_shares = np.concatenate((np.tile(0.0, 2), type_shares), axis=0)

    # Type shifts
    type_shifts = x[50 + (num_types - 1) * 2:]
    type_shifts = np.reshape(type_shifts, (num_types - 1, 4))
    type_shifts = np.concatenate((np.tile(0.0, (1, 4)), type_shifts), axis=0)

    return type_shares, type_shifts


def extract_cholesky(x, info=None):
    """ Construct the Cholesky matrix.
    """
    shocks_cholesky = np.tile(0.0, (4, 4))
    shocks_cholesky[0, :1] = x[40:41]
    shocks_cholesky[1, :2] = x[41:43]
    shocks_cholesky[2, :3] = x[43:46]
    shocks_cholesky[3, :4] = x[46:50]

    # Stabilization
    if info is not None:
        info = 0

    # We need to ensure that the diagonal elements are larger than zero during an estimation.
    # However, we want to allow for the special case of total absence of randomness for testing
    # purposes of simulated datasets.
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


def get_total_values(period, num_periods, optim_paras, rewards_systematic, draws, edu_spec,
                     mapping_state_idx, periods_emax, k, states_all):
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
        emaxs = get_emaxs(edu_spec, mapping_state_idx, period, periods_emax, k, states_all)
    else:
        emaxs = np.tile(0.0, 4)

    # Calculate total utilities
    total_values = rewards_ex_post + optim_paras['delta'] * emaxs

    # This is required to ensure that the agent does not choose any inadmissible states. If the
    # state is inadmissible emaxs takes value zero.
    if states_all[period, k, 2] >= edu_spec['max']:
        total_values[2] += INADMISSIBILITY_PENALTY

    # Finishing
    return total_values


def get_emaxs(edu_spec, mapping_state_idx, period, periods_emax, k, states_all):
    """ Get emaxs for additional choices.
    """
    # Distribute state space
    exp_a, exp_b, edu, _, type_ = states_all[period, k, :]

    # Future utilities
    emaxs = np.tile(np.nan, 4)

    # Working in Occupation A
    future_idx = mapping_state_idx[period + 1, exp_a + 1, exp_b, edu, 2, type_]
    emaxs[0] = periods_emax[period + 1, future_idx]

    # Working in Occupation B
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b + 1, edu, 3, type_]
    emaxs[1] = periods_emax[period + 1, future_idx]

    # Increasing schooling. Note that adding an additional year of schooling is only possible for
    # those that have strictly less than the maximum level of additional education allowed.
    is_inadmissible = (edu >= edu_spec['max'])
    if is_inadmissible:
        emaxs[2] = 0.00
    else:
        future_idx = mapping_state_idx[period + 1, exp_a, exp_b, edu + 1, 1, type_]
        emaxs[2] = periods_emax[period + 1, future_idx]

    # Staying at home
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b, edu, 0, type_]
    emaxs[3] = periods_emax[period + 1, future_idx]

    # Finishing
    return emaxs


def create_draws(num_periods, num_draws, seed, is_debug):
    """ Create the relevant set of draws. Handle special case of zero variances as thi case is 
    useful for hand-based testing. The draws are drawn from a standard normal distribution and 
    transformed later in the code.
    """
    # Control randomness by setting seed value
    np.random.seed(seed)

    # Draw random deviates from a standard normal distribution or read it in from disk. The
    # latter is available to allow for testing across implementation.
    if is_debug and os.path.exists('.draws.respy.test'):
        draws = read_draws(num_periods, num_draws)
    else:
        draws = np.random.multivariate_normal(np.zeros(4), np.identity(4), (num_periods, num_draws))

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


def add_solution(respy_obj, periods_rewards_systematic, states_number_period, mapping_state_idx,
                 periods_emax, states_all):
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
    # Auxiliary objects
    num_types = len(optim_paras['type_shifts'])

    # Checks for all arguments
    keys = []
    keys += ['coeffs_a', 'coeffs_b', 'coeffs_edu', 'coeffs_home', 'level', 'shocks_cholesky']
    keys += ['delta', 'type_shares', 'type_shifts', 'coeffs_common']

    for key in keys:
        assert (isinstance(optim_paras[key], np.ndarray))
        assert (np.all(np.isfinite(optim_paras[key])))
        assert (optim_paras[key].dtype == 'float')
        assert (np.all(abs(optim_paras[key]) < PRINT_FLOAT))

    # Check for discount rate
    assert (optim_paras['delta'] >= 0)

    # Check for level of ambiguity
    assert (optim_paras['level'] >= 0)

    # Checks for common returns
    assert (optim_paras['coeffs_common'].size == 2)

    # Checks for occupations
    assert (optim_paras['coeffs_a'].size == 13)
    assert (optim_paras['coeffs_b'].size == 13)
    assert (optim_paras['coeffs_edu'].size == 7)
    assert (optim_paras['coeffs_home'].size == 3)

    # Checks shock matrix
    assert (optim_paras['shocks_cholesky'].shape == (4, 4))
    np.allclose(optim_paras['shocks_cholesky'], np.tril(optim_paras['shocks_cholesky']))

    # Checks for type shares
    assert optim_paras['type_shares'].size == num_types * 2

    # Checks for type shifts
    assert optim_paras['type_shifts'].shape == (num_types, 4)

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
    draws = np.array(np.genfromtxt('.draws.respy.test'), ndmin=2)
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


def generate_optimizer_options(which, optim_paras, num_paras):

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

        # It is not recommended that N is larger than upper as the code might break down due to a
        # segmentation fault. See the source files for the absolute upper bounds.
        assert sum(optim_paras['paras_fixed']) != num_paras
        lower = (num_paras - sum(optim_paras['paras_fixed'])) + 2
        upper = (2 * (num_paras - sum(optim_paras['paras_fixed'])) + 1)
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
    """ Print initialization dictionary to file. The different formatting makes the file rather 
    involved. The resulting initialization files are rad by PYTHON and FORTRAN routines. Thus, 
    the formatting with respect to the number of decimal places is rather small.
    """
    assert (isinstance(dict_, dict))

    opt_labels = []
    opt_labels += ['BASICS', 'AMBIGUITY', 'COMMON', 'OCCUPATION A', 'OCCUPATION B', 'EDUCATION']
    opt_labels += ['HOME', 'SHOCKS', 'TYPE SHARES', 'TYPE SHIFTS']

    str_optim = '{0:<10} {1:25.15f} {2:>5} {3:>15}\n'

    # Construct labels. This ensures that the initialization files always look identical.
    labels = opt_labels
    labels += ['SOLUTION', 'SIMULATION', 'ESTIMATION', 'DERIVATIVES', 'PRECONDITIONING']
    labels += ['PROGRAM', 'INTERPOLATION']
    labels += OPT_EST_FORT + OPT_EST_PYTH + ['SCIPY-SLSQP', 'FORT-SLSQP']

    num_types = int(len(dict_['TYPE SHARES']['coeffs']) / 2) + 1

    # Create initialization.
    with open(file_name, 'w') as file_:

        for flag in labels:
            if flag in ['BASICS']:

                file_.write('BASICS\n\n')

                str_ = '{0:<10} {1:>25}\n'
                file_.write(str_.format('periods', dict_[flag]['periods']))

                line = format_opt_parameters(dict_['BASICS'], 0)
                file_.write(str_optim.format(*line))

                file_.write('\n')

            if flag in ['TYPE SHARES'] and num_types > 1:
                file_.write(flag.upper() + '\n\n')

                for i in range(num_types - 1):
                    for j in range(2):
                        pos = (i * 2) + j
                        line = format_opt_parameters(dict_['TYPE SHARES'], pos)
                        file_.write(str_optim.format(*line))
                    file_.write('\n')

            if flag in ['TYPE SHIFTS'] and num_types > 1:
                file_.write(flag.upper() + '\n\n')

                for i in range(num_types - 1):
                    for j in range(4):
                        pos = (i * 4) + j
                        line = format_opt_parameters(dict_['TYPE SHIFTS'], pos)
                        file_.write(str_optim.format(*line))
                    file_.write('\n')

            if flag in ['HOME']:

                file_.write(flag.upper() + '\n\n')
                for i in range(3):
                    line = format_opt_parameters(dict_['HOME'], i)
                    file_.write(str_optim.format(*line))

                file_.write('\n')

            if flag in ['SOLUTION', 'SIMULATION', 'PROGRAM', 'INTERPOLATION',
                        'ESTIMATION', 'PRECONDITIONING', 'DERIVATIVES']:

                file_.write(flag.upper() + '\n\n')
                keys = list(dict_[flag].keys())
                keys.sort()
                for key_ in keys:

                    if key_ in ['tau']:
                        str_ = '{0:<10} {1:25.15f}\n'
                        file_.write(str_.format(key_, dict_[flag][key_]))
                    else:
                        str_ = '{0:<10} {1:>25}\n'
                        file_.write(str_.format(key_, str(dict_[flag][key_])))

                file_.write('\n')

            if flag in ['SHOCKS']:

                # Type conversion
                file_.write(flag.upper() + '\n\n')
                for i in range(10):
                    line = format_opt_parameters(dict_['SHOCKS'], i)
                    file_.write(str_optim.format(*line))
                file_.write('\n')

            if flag in ['EDUCATION']:

                file_.write(flag.upper() + '\n\n')

                for i in range(7):
                    line = format_opt_parameters(dict_['EDUCATION'], i)
                    file_.write(str_optim.format(*line))

                file_.write('\n')
                str_ = '{0:<10} {1:>25}\n'
                for i, start in enumerate(dict_[flag]['start']):
                    file_.write(str_.format('start', start))
                    file_.write(str_.format('share', dict_[flag]['share'][i]))
                    file_.write('\n')

                file_.write(str_.format('max', dict_[flag]['max']))

                file_.write('\n')

            if flag in ['AMBIGUITY']:
                file_.write(flag.upper() + '\n\n')
                line = format_opt_parameters(dict_['AMBIGUITY'], 0)
                file_.write(str_optim.format(*line))

                str_ = '{0:<10} {1:>25}\n'
                file_.write(str_.format('measure', dict_[flag]['measure']))
                file_.write(str_.format('mean', str(dict_[flag]['mean'])))

                file_.write('\n')

            if flag in ['COMMON']:
                file_.write(flag + '\n\n')
                for j in range(2):
                    line = format_opt_parameters(dict_[flag], j)
                    file_.write(str_optim.format(*line))
                file_.write('\n')

            if flag in ['OCCUPATION A', 'OCCUPATION B']:
                file_.write(flag + '\n\n')
                for j in range(13):
                    line = format_opt_parameters(dict_[flag], j)
                    file_.write(str_optim.format(*line))
                    # Visual separation of the parameters from the skill function.
                    if j == 9:
                        file_.write('\n')


                file_.write('\n')

            if flag in OPTIMIZERS:

                # This function can also be used to print out initialization files without any
                # optimization options. This is enough for simulation tasks.
                if flag not in dict_.keys():
                    continue

                file_.write(flag.upper() + '\n\n')
                keys = list(dict_[flag].keys())
                keys.sort()
                for key_ in keys:

                    if key_ in ['maxfun', 'npt', 'maxiter', 'm', 'maxls']:
                        str_ = '{0:<10} {1:>25}\n'
                        file_.write(str_.format(key_, dict_[flag][key_]))
                    else:
                        str_ = '{0:<10} {1:25.15f}\n'
                        file_.write(str_.format(key_, dict_[flag][key_]))

                file_.write('\n')


def format_opt_parameters(dict_, pos):
    """ This function formats the values depending on whether they are fixed
    during the optimization or not.
    """
    # Initialize baseline line
    val = dict_['coeffs'][pos]
    is_fixed = dict_['fixed'][pos]
    bounds = dict_['bounds'][pos]

    line = ['coeff', val, ' ', ' ']
    if is_fixed:
        line[-2] = '!'

    # Check if any bounds defined
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
    line = shlex.split(linecache.getline('est.respy.info', 46))
    rslt['num_step'] = _process_value(line[3], 'int')

    line = shlex.split(linecache.getline('est.respy.info', 48))
    rslt['num_eval'] = _process_value(line[3], 'int')

    # Parameter values
    for i, key_ in enumerate(['start', 'step', 'current']):
        rslt['paras_' + key_] = []
        for j in range(13, 100):
            line = shlex.split(linecache.getline('est.respy.info', j))
            if not line:
                break
            rslt['paras_' + key_] += [_process_value(line[i + 1], 'float')]
        rslt['paras_' + key_] = np.array(rslt['paras_' + key_])

    return rslt


def remove_scratch(fname):
    """ This function removes scratch files.
    """
    if os.path.exists(fname):
        os.unlink(fname)


def get_optim_paras(optim_paras, num_paras, which, is_debug):
    """ Get optimization parameters.
    """
    # Checks
    if is_debug:
        assert check_model_parameters(optim_paras)

    # Auxiliary objects
    num_types = len(optim_paras['type_shifts'])

    # Initialize container
    x = np.tile(np.nan, num_paras)

    # Discount rate
    x[0:1] = optim_paras['delta']

    # Level of Ambiguity
    x[1:2] = optim_paras['level']

    # Occupation A
    x[2:4] = optim_paras['coeffs_common']

    # Occupation A
    x[4:17] = optim_paras['coeffs_a']

    # Occupation B
    x[17:30] = optim_paras['coeffs_b']

    # Education
    x[30:37] = optim_paras['coeffs_edu']

    # Home
    x[37:40] = optim_paras['coeffs_home']

    # Shocks
    x[40:50] = optim_paras['shocks_cholesky'][np.tril_indices(4)]

    # Shares
    x[50:50 + (num_types - 1) * 2] = optim_paras['type_shares'][2:]

    x[50 + (num_types - 1) * 2:num_paras] = optim_paras['type_shifts'].flatten()[4:]

    # Checks
    if is_debug:
        check_optimization_parameters(x)

    # Select subset
    if which == 'free':
        x_free_curre = []
        for i in range(num_paras):
            if not optim_paras['paras_fixed'][i]:
                x_free_curre += [x[i]]

        x = np.array(x_free_curre)

    # Finishing
    return x


def covariance_to_correlation(cov):
    """ This function constructs the correlation matrix from the information on the covariances.
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
    """ This function constructs the covariance matrix from the information on the correlations.
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
    """ This function checks for reasons that require an early termination of the optimization 
    procedure.
    """
    # We want an early termination if the number of function evaluations is already at the
    # maximum number requested. This is not strictly enforced in some of the SCIPY algorithms.
    if maxfun == num_eval:
        raise MaxfunError

    # We also want the opportunity for an immediate, but gentle termination from the user.
    if os.path.exists('.stop.respy.scratch'):
        raise MaxfunError


def get_num_obs_agent(data_array, num_agents_est):
    """ Get a list with the number of observations for each agent. 
    """
    num_obs_agent = np.tile(0, num_agents_est)
    agent_number = data_array[0, 0]
    num_rows = data_array.shape[0]

    q = 0
    for i in range(num_rows):
        # We need to check whether we are faced with a new agent.
        if data_array[i, 0] != agent_number:
            q += 1
            agent_number = data_array[i, 0]

        num_obs_agent[q] += 1

    return num_obs_agent


def back_out_systematic_wages(rewards_systematic, exp_a, exp_b, edu, activity_lagged, optim_paras):
    """ This function constructs the wage component for the labor market rewards.
    """
    # Construct covariates to construct the general component of labor market rewards.
    covariates = construct_covariates(exp_a, exp_b, edu, activity_lagged, None, None)

    # First we calculate the general component.
    general, wages_systematic = np.tile(np.nan, 2), np.tile(np.nan, 2)

    covars_general = [1.0, covariates['not_exp_a_lagged'], covariates['not_any_exp_a']]
    general[0] = np.dot(optim_paras['coeffs_a'][10:], covars_general)

    covars_general = [1.0, covariates['not_exp_b_lagged'], covariates['not_any_exp_b']]
    general[1] = np.dot(optim_paras['coeffs_b'][10:], covars_general)

    # Second we do the same with the common component.
    covars_common = [covariates['hs_graduate'], covariates['co_graduate']]
    rewards_common = np.dot(optim_paras['coeffs_common'], covars_common)

    for j in [0, 1]:
        wages_systematic[j] = rewards_systematic[j] - general[j] - rewards_common

    return wages_systematic


def construct_covariates(exp_a, exp_b, edu, activity_lagged, type_, period):
    """ Construction of some additional covariates for the reward calculations.
    """
    covariates = dict()
    covariates['not_exp_a_lagged'] = int(activity_lagged != 2)
    covariates['not_exp_b_lagged'] = int(activity_lagged != 3)
    covariates['edu_lagged'] = int(activity_lagged == 1)
    covariates['not_any_exp_a'] = int(exp_a == 0)
    covariates['not_any_exp_b'] = int(exp_b == 0)
    covariates['activity_lagged'] = activity_lagged
    covariates['period'] = period
    covariates['exp_a'] = exp_a
    covariates['exp_b'] = exp_b
    covariates['type'] = type_
    covariates['edu'] = edu

    if edu is not None:
        covariates['hs_graduate'] = int(edu >= 12)
        covariates['co_graduate'] = int(edu >= 16)

        cond = (not covariates['edu_lagged']) and (not covariates['hs_graduate'])
        covariates['is_return_not_high_school'] = int(cond)

        cond = (not covariates['edu_lagged']) and covariates['hs_graduate']
        covariates['is_return_high_school'] = int(cond)

    if period is not None:
        covariates['is_minor'] = int(period < 2)

    return covariates