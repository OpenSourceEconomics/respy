"""Process a Respy initialization file.

The module contains tools to read and write initialization files and to
convert between two different dictionary representations of a model:

1) init_dict: a dictionary that contains all of the information from the .ini file
in a structure that is very close to that file.

2) attr_dict: a dictionary that contains the information from the .ini file and
additional derived information in a structure that is helpful for further
usage in RespyCLS.

"""
import numpy as np
import shlex
from respy.python.shared.shared_constants import OPT_EST_FORT
from respy.python.shared.shared_constants import OPT_EST_PYTH
from respy.python.shared.shared_auxiliary import cholesky_to_coeffs
from respy.python.shared.shared_auxiliary import format_opt_parameters
from respy.python.shared.shared_auxiliary import cholcov_from_econ_coeffs
from respy.pre_processing.model_processing_auxiliary import \
    _process_coefficient_line, _type_conversions, _add_to_dictionary, \
    _determine_case, _paras_mapping


def read_init_file(fname):
    """Read a RESPY initialization file into a Python dictionary.

    The design of the initialization files is optimized for readability and user
    friendliness, not developer friendliness.

    The resulting dictionary is very close to the structure of the ini
    file and will be processed later.

    """
    dict_ = {}

    with open(fname) as in_file:
        for line in in_file.readlines():
            line_entries = shlex.split(line)
            is_empty, is_group, is_comment = _determine_case(line_entries)

            if is_empty or is_comment:
                continue

            if is_group:
                group = line_entries[0]
                if group in ['OCCUPATION', 'TYPE']:
                    group = line_entries[0] + ' ' + line_entries[1]
                dict_[group] = {}
                continue

            # is the block for the optimization coefficients.
            flag, value = line_entries[:2]
            is_fixed, bounds = None, None

            if flag in ['coeff']:
                is_fixed, bounds = _process_coefficient_line(line)

            value = _type_conversions(flag, value)

            _add_to_dictionary(group, flag, value, is_fixed, bounds, dict_)

    dict_['SHOCKS']['coeffs'] = np.array(dict_['SHOCKS']['coeffs'])

    init_dict = default_model_dict()
    init_dict.update(dict_)

    return init_dict


def write_init_file(init_dict, file_name='test.respy.ini'):
    """Write initialization file with information from init_dict.

    The different formatting makes the file rather involved.
    The resulting initialization files are rad by PYTHON and FORTRAN routines.
    Thus, the formatting with respect to the number of decimal places is rather
    small.

    """
    assert (isinstance(init_dict, dict))

    opt_labels = ['BASICS', 'COMMON', 'OCCUPATION A', 'OCCUPATION B',
                  'EDUCATION', 'HOME', 'SHOCKS', 'TYPE SHARES', 'TYPE SHIFTS']

    str_optim = '{0:<10} {1:25.15f} {2:>5} {3:>15}\n'

    # Construct labels to ensure that the ini files always look identical.
    labels = opt_labels
    labels += ['SOLUTION', 'SIMULATION', 'ESTIMATION', 'DERIVATIVES',
               'PRECONDITIONING', 'PROGRAM', 'INTERPOLATION']
    labels += OPT_EST_FORT + OPT_EST_PYTH

    num_types = int(len(init_dict['TYPE SHARES']['coeffs']) / 2) + 1

    # Create initialization.
    with open(file_name, 'w') as file_:

        for flag in labels:
            if flag in ['BASICS']:

                file_.write('BASICS\n\n')

                str_ = '{0:<10} {1:>25}\n'
                file_.write(str_.format('periods', init_dict[flag]['periods']))

                line = format_opt_parameters(init_dict['BASICS'], 0)
                file_.write(str_optim.format(*line))

                file_.write('\n')

            if flag in ['TYPE SHARES'] and num_types > 1:
                file_.write(flag.upper() + '\n\n')

                for i in range(num_types - 1):
                    for j in range(2):
                        pos = (i * 2) + j
                        line = format_opt_parameters(init_dict['TYPE SHARES'], pos)
                        file_.write(str_optim.format(*line))
                    file_.write('\n')

            if flag in ['TYPE SHIFTS'] and num_types > 1:
                file_.write(flag.upper() + '\n\n')

                for i in range(num_types - 1):
                    for j in range(4):
                        pos = (i * 4) + j
                        line = format_opt_parameters(init_dict['TYPE SHIFTS'], pos)
                        file_.write(str_optim.format(*line))
                    file_.write('\n')

            if flag in ['HOME']:

                file_.write(flag.upper() + '\n\n')
                for i in range(3):
                    line = format_opt_parameters(init_dict['HOME'], i)
                    file_.write(str_optim.format(*line))

                file_.write('\n')

            if flag in ['SOLUTION', 'SIMULATION', 'PROGRAM', 'INTERPOLATION',
                        'ESTIMATION', 'PRECONDITIONING', 'DERIVATIVES']:

                file_.write(flag.upper() + '\n\n')
                keys = list(init_dict[flag].keys())
                keys.sort()
                for key_ in keys:

                    if key_ in ['tau']:
                        str_ = '{0:<10} {1:25.15f}\n'
                        file_.write(str_.format(key_, init_dict[flag][key_]))
                    else:
                        str_ = '{0:<10} {1:>25}\n'
                        file_.write(str_.format(key_, str(init_dict[flag][key_])))

                file_.write('\n')

            if flag in ['SHOCKS']:

                # Type conversion
                file_.write(flag.upper() + '\n\n')
                for i in range(10):
                    line = format_opt_parameters(init_dict['SHOCKS'], i)
                    file_.write(str_optim.format(*line))
                file_.write('\n')

            if flag in ['EDUCATION']:

                file_.write(flag.upper() + '\n\n')

                for i in range(7):
                    line = format_opt_parameters(init_dict['EDUCATION'], i)
                    file_.write(str_optim.format(*line))

                file_.write('\n')
                str_ = '{0:<10} {1:>25}\n'
                for i, start in enumerate(init_dict[flag]['start']):
                    file_.write(str_.format('start', start))
                    file_.write(str_.format('share', init_dict[flag]['share'][i]))
                    file_.write(
                        str_.format('lagged', init_dict[flag]['lagged'][i]))

                    file_.write('\n')

                file_.write(str_.format('max', init_dict[flag]['max']))

                file_.write('\n')

            if flag in ['COMMON']:
                file_.write(flag + '\n\n')
                for j in range(2):
                    line = format_opt_parameters(init_dict[flag], j)
                    file_.write(str_optim.format(*line))
                file_.write('\n')

            if flag in ['OCCUPATION A', 'OCCUPATION B']:
                file_.write(flag + '\n\n')
                for j in range(15):
                    line = format_opt_parameters(init_dict[flag], j)
                    file_.write(str_optim.format(*line))
                    # Visual separation of parameters from skill function.
                    if j == 11:
                        file_.write('\n')

                file_.write('\n')

            if flag in OPT_EST_FORT + OPT_EST_PYTH:

                # This function can also be used to print out initialization
                # files without optimization options (enough for simulation).
                if flag not in init_dict.keys():
                    continue

                file_.write(flag.upper() + '\n\n')
                keys = list(init_dict[flag].keys())
                keys.sort()
                for key_ in keys:

                    if key_ in ['maxfun', 'npt', 'maxiter', 'm', 'maxls']:
                        str_ = '{0:<10} {1:>25}\n'
                        file_.write(str_.format(key_, init_dict[flag][key_]))
                    else:
                        str_ = '{0:<10} {1:25.15f}\n'
                        file_.write(str_.format(key_, init_dict[flag][key_]))

                file_.write('\n')


def convert_init_dict_to_attr_dict(init_dict):
    """Convert an init_dict to attr_dict."""
    attr = {}
    ini = init_dict
    attr['num_points_interp'] = ini['INTERPOLATION']['points']
    attr['optimizer_used'] = ini['ESTIMATION']['optimizer']
    attr['is_interpolated'] = ini['INTERPOLATION']['flag']
    attr['num_agents_sim'] = ini['SIMULATION']['agents']
    attr['num_agents_est'] = ini['ESTIMATION']['agents']
    attr['derivatives'] = ini['DERIVATIVES']['version']
    attr['num_draws_prob'] = ini['ESTIMATION']['draws']
    attr['num_draws_emax'] = ini['SOLUTION']['draws']
    attr['num_periods'] = ini['BASICS']['periods']
    attr['seed_prob'] = ini['ESTIMATION']['seed']
    attr['maxfun'] = ini['ESTIMATION']['maxfun']
    attr['seed_sim'] = ini['SIMULATION']['seed']
    attr['file_sim'] = ini['SIMULATION']['file']
    attr['file_est'] = ini['ESTIMATION']['file']
    attr['is_store'] = ini['SOLUTION']['store']
    attr['seed_emax'] = ini['SOLUTION']['seed']
    attr['version'] = ini['PROGRAM']['version']
    attr['num_procs'] = ini['PROGRAM']['procs']
    attr['is_debug'] = ini['PROGRAM']['debug']
    attr['edu_max'] = ini['EDUCATION']['max']
    attr['tau'] = ini['ESTIMATION']['tau']
    attr['precond_spec'] = {}
    attr['precond_spec']['minimum'] = ini['PRECONDITIONING']['minimum']
    attr['precond_spec']['type'] = ini['PRECONDITIONING']['type']
    attr['precond_spec']['eps'] = ini['PRECONDITIONING']['eps']
    attr['edu_spec'] = {}
    attr['edu_spec']['lagged'] = ini['EDUCATION']['lagged']
    attr['edu_spec']['start'] = ini['EDUCATION']['start']
    attr['edu_spec']['share'] = ini['EDUCATION']['share']
    attr['edu_spec']['max'] = ini['EDUCATION']['max']

    attr['optim_paras'] = {}

    # Constructing the covariance matrix of the shocks
    shocks_coeffs = ini['SHOCKS']['coeffs']
    attr['optim_paras']['shocks_cholesky'] = cholcov_from_econ_coeffs(shocks_coeffs)

    # Constructing the shifts for each type.
    type_shifts = ini['TYPE SHIFTS']['coeffs']
    type_shares = ini['TYPE SHARES']['coeffs']

    # TO-DO: num_types is a derived attribute
    attr['num_types'] = int(len(ini['TYPE SHARES']['coeffs']) / 2) + 1

    if attr['num_types'] == 1:
        type_shares = np.tile(0.0, 2)
        type_shifts = np.tile(0.0, (1, 4))
    else:
        type_shares = np.concatenate((np.tile(0.0, 2), type_shares), axis=0)
        type_shifts = np.reshape(type_shifts, (attr['num_types'] - 1, 4))
        type_shifts = np.concatenate((np.tile(0.0, (1, 4)), type_shifts), axis=0)

    attr['optim_paras']['type_shifts'] = type_shifts
    attr['optim_paras']['type_shares'] = type_shares

    attr['optim_paras']['coeffs_a'] = ini['OCCUPATION A']['coeffs']
    attr['optim_paras']['coeffs_b'] = ini['OCCUPATION B']['coeffs']
    attr['optim_paras']['coeffs_common'] = ini['COMMON']['coeffs']
    attr['optim_paras']['coeffs_edu'] = ini['EDUCATION']['coeffs']
    attr['optim_paras']['coeffs_home'] = ini['HOME']['coeffs']
    attr['optim_paras']['delta'] = ini['BASICS']['coeffs']

    # Initialize information about optimization parameters
    keys = ['BASICS', 'COMMON', 'OCCUPATION A', 'OCCUPATION B',
            'EDUCATION', 'HOME', 'SHOCKS', 'TYPE SHARES', 'TYPE SHIFTS']

    for which in ['fixed', 'bounds']:
        attr['optim_paras']['paras_' + which] = []
        for key_ in keys:
            attr['optim_paras']['paras_' + which] += ini[key_][which][:]

    # Ensure that all elements in the dictionary are of the same type.
    keys = ['coeffs_a', 'coeffs_b', 'coeffs_edu', 'coeffs_home',
            'shocks_cholesky', 'delta', 'type_shares', 'type_shifts',
            'coeffs_common']
    for key_ in keys:
        attr['optim_paras'][key_] = \
            np.array(attr['optim_paras'][key_])

    # Aggregate all the information provided about optimizer options in
    # one class attribute for easier access later.
    attr['optimizer_options'] = dict()
    for optimizer in OPT_EST_FORT + OPT_EST_PYTH:
        is_defined = (optimizer in ini.keys())
        if is_defined:
            attr['optimizer_options'][optimizer] = ini[optimizer]

    # We need to align the indicator for the fixed parameters.
    # In the initialization file, these refer to the upper triangular
    # matrix of the covariances. Inside the  program, we use the lower
    # triangular Cholesky decomposition.
    paras_fixed = attr['optim_paras']['paras_fixed'][:]

    paras_fixed_reordered = paras_fixed[:]

    for old, new in _paras_mapping():
        paras_fixed_reordered[new] = paras_fixed[old]

    attr['optim_paras']['paras_fixed'] = paras_fixed_reordered

    return attr


def convert_attr_dict_to_init_dict(attr_dict):
    """Convert an attr_dict to init_dict."""
    # TO-DO: remove hard coded parameter positions
    ini = {}
    attr = attr_dict
    num_paras = attr['num_paras']
    num_types = attr['num_types']

    # Basics
    ini['BASICS'] = {}
    ini['BASICS']['periods'] = attr['num_periods']
    ini['BASICS']['coeffs'] = attr['optim_paras']['delta']
    ini['BASICS']['bounds'] = attr['optim_paras']['paras_bounds'][0:1]
    ini['BASICS']['fixed'] = attr['optim_paras']['paras_fixed'][0:1]

    # Common Returns
    lower, upper = 1, 3
    ini['COMMON'] = {}
    ini['COMMON']['coeffs'] = attr['optim_paras']['coeffs_common']
    ini['COMMON']['bounds'] = attr['optim_paras']['paras_bounds'][lower:upper]
    ini['COMMON']['fixed'] = attr['optim_paras']['paras_fixed'][lower:upper]

    # Occupation A
    lower, upper = 3, 18
    ini['OCCUPATION A'] = {}
    ini['OCCUPATION A']['coeffs'] = attr['optim_paras']['coeffs_a']

    ini['OCCUPATION A']['bounds'] = attr['optim_paras']['paras_bounds'][lower:upper]
    ini['OCCUPATION A']['fixed'] = attr['optim_paras']['paras_fixed'][lower:upper]

    # Occupation B
    lower, upper = 18, 33
    ini['OCCUPATION B'] = {}
    ini['OCCUPATION B']['coeffs'] = attr['optim_paras']['coeffs_b']

    ini['OCCUPATION B']['bounds'] = attr['optim_paras']['paras_bounds'][lower:upper]
    ini['OCCUPATION B']['fixed'] = attr['optim_paras']['paras_fixed'][lower:upper]

    # Education
    lower, upper = 33, 40
    ini['EDUCATION'] = {}
    ini['EDUCATION']['coeffs'] = attr['optim_paras']['coeffs_edu']

    ini['EDUCATION']['bounds'] = attr['optim_paras']['paras_bounds'][lower:upper]
    ini['EDUCATION']['fixed'] = attr['optim_paras']['paras_fixed'][lower:upper]

    ini['EDUCATION']['lagged'] = attr['edu_spec']['lagged']
    ini['EDUCATION']['start'] = attr['edu_spec']['start']
    ini['EDUCATION']['share'] = attr['edu_spec']['share']
    ini['EDUCATION']['max'] = attr['edu_spec']['max']

    # Home
    lower, upper = 40, 43
    ini['HOME'] = {}
    ini['HOME']['coeffs'] = attr['optim_paras']['coeffs_home']

    ini['HOME']['bounds'] = attr['optim_paras']['paras_bounds'][lower:upper]
    ini['HOME']['fixed'] = attr['optim_paras']['paras_fixed'][lower:upper]

    # Shocks
    lower, upper = 43, 53
    ini['SHOCKS'] = {}
    shocks_cholesky = attr['optim_paras']['shocks_cholesky']
    shocks_coeffs = cholesky_to_coeffs(shocks_cholesky)
    ini['SHOCKS']['coeffs'] = shocks_coeffs

    ini['SHOCKS']['bounds'] = attr['optim_paras']['paras_bounds'][lower:upper]

    # Again we need to reorganize the order of the coefficients
    paras_fixed_reordered = attr['optim_paras']['paras_fixed'][:]

    paras_fixed = paras_fixed_reordered[:]
    for old, new in _paras_mapping():
        paras_fixed[old] = paras_fixed_reordered[new]

    ini['SHOCKS']['fixed'] = paras_fixed[43:53]

    # Solution
    ini['SOLUTION'] = {}
    ini['SOLUTION']['draws'] = attr['num_draws_emax']
    ini['SOLUTION']['seed'] = attr['seed_emax']
    ini['SOLUTION']['store'] = attr['is_store']

    # Type Shares
    lower, upper = 53, 53 + (num_types - 1) * 2
    ini['TYPE SHARES'] = {}
    ini['TYPE SHARES']['coeffs'] = attr['optim_paras']['type_shares'][2:]
    ini['TYPE SHARES']['bounds'] = attr['optim_paras']['paras_bounds'][lower:upper]
    ini['TYPE SHARES']['fixed'] = attr['optim_paras']['paras_fixed'][lower:upper]

    # Type Shifts
    lower, upper = 53 + (num_types - 1) * 2, num_paras
    ini['TYPE SHIFTS'] = {}
    ini['TYPE SHIFTS']['coeffs'] = attr['optim_paras']['type_shifts'].flatten()[4:]
    ini['TYPE SHIFTS']['bounds'] = attr['optim_paras']['paras_bounds'][lower:upper]
    ini['TYPE SHIFTS']['fixed'] = attr['optim_paras']['paras_fixed'][lower:upper]

    # Simulation
    ini['SIMULATION'] = {}
    ini['SIMULATION']['agents'] = attr['num_agents_sim']
    ini['SIMULATION']['file'] = attr['file_sim']
    ini['SIMULATION']['seed'] = attr['seed_sim']

    # Estimation
    ini['ESTIMATION'] = {}
    ini['ESTIMATION']['optimizer'] = attr['optimizer_used']
    ini['ESTIMATION']['agents'] = attr['num_agents_est']
    ini['ESTIMATION']['draws'] = attr['num_draws_prob']
    ini['ESTIMATION']['seed'] = attr['seed_prob']
    ini['ESTIMATION']['file'] = attr['file_est']
    ini['ESTIMATION']['maxfun'] = attr['maxfun']
    ini['ESTIMATION']['tau'] = attr['tau']

    # Derivatives
    ini['DERIVATIVES'] = {}
    ini['DERIVATIVES']['version'] = attr['derivatives']

    # Scaling
    ini['PRECONDITIONING'] = {}
    ini['PRECONDITIONING']['minimum'] = \
        attr['precond_spec']['minimum']
    ini['PRECONDITIONING']['type'] = \
        attr['precond_spec']['type']
    ini['PRECONDITIONING']['eps'] = attr['precond_spec']['eps']

    # Program
    ini['PROGRAM'] = {}
    ini['PROGRAM']['version'] = attr['version']
    ini['PROGRAM']['procs'] = attr['num_procs']
    ini['PROGRAM']['debug'] = attr['is_debug']

    # Interpolation
    ini['INTERPOLATION'] = {}
    ini['INTERPOLATION']['points'] = attr['num_points_interp']
    ini['INTERPOLATION']['flag'] = attr['is_interpolated']

    # Optimizers
    for optimizer in attr['optimizer_options'].keys():
        ini[optimizer] = attr['optimizer_options'][optimizer]

    return ini


def default_model_dict():
    """Return a partial init_dict with default values.

    This is not a complete init_dict. It only contains the parts
    for which default values make sense.

    """
    default = {
        'TYPE SHARES': {'coeffs': [], 'fixed': [], 'bounds': []},
        'TYPE SHIFTS': {'coeffs': [], 'fixed': [], 'bounds': []},
        'FORT_NEWUOA': {'maxfun': 1000000, 'npt': 1, 'rhobeg': 1.0, 'rhoend': 0.000001},
        'FORT-BFGS': {'eps': 0.0001, 'gtol': 0.00001, 'maxiter': 10, 'stpmx': 100.0},
        'FORT-BOBYQA': {'maxfun': 1000000, 'npt': 1, 'rhobeg': 1.0, 'rhoend': 0.000001},
        'SCIPY-BFGS': {'eps': 0.0001, 'gtol': 0.0001, 'maxiter': 1},
        'SCIPY-POWELL': {'ftol': 0.0001, 'maxfun': 1000000, 'maxiter': 1, 'xtol': 0.0001},
        'SCIPY-LBFGSB': {'eps': 0.000000441037423, 'factr': 30.401091854739622, 'm': 5,
                         'maxiter': 2, 'maxls': 2, 'pgtol': 0.000086554171164}
    }

    return default
