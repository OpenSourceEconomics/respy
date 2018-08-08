"""Process a Respy initialization file.



"""
import numpy as np
import shlex
from respy.python.shared.shared_constants import OPT_EST_FORT
from respy.python.shared.shared_constants import OPT_EST_PYTH
from respy.python.shared.shared_auxiliary import cholesky_to_coeffs


OPTIMIZERS = OPT_EST_FORT + OPT_EST_PYTH

# We need to do some reorganization as the parameters from the
# initialization that describe the covariance of the shocks need to be
# mapped to the Cholesky factors which are the parameters the optimizer
# actually iterates on.
PARAS_MAPPING = [
    (43, 43), (44, 44), (45, 46), (46, 49), (47, 45), (48, 47),
    (49, 50), (50, 48), (51, 51), (52, 52)]


def read_and_process_ini_file(fname):
    ini = read(fname)
    attr = convert_init_dict_to_attr_dict(ini)
    return attr


def read(fname):
    """Read and aRESPY initialization file into a Python dictionary.

    The design of the initialization files is optimized for readability and user
    friendliness, not developer friendliness.

    The resulting dictionary is still very close to the structure of the ini
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
                is_fixed, bounds = process_coefficient_line(line)

            value = _type_conversions(flag, value)

            _add_to_dictionary(group, flag, value, is_fixed, bounds, dict_)

    dict_['SHOCKS']['coeffs'] = np.array(dict_['SHOCKS']['coeffs'])

    init_dict = default_model_dict()
    init_dict.update(dict_)

    return init_dict


def convert_init_dict_to_attr_dict(init_dict):
    """Convert init_dict to an attribute dictionary of RespyCls."""
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
    for i in [0, 4, 7, 9]:
        shocks_coeffs[i] **= 2

    shocks = np.zeros((4, 4))
    shocks[0, :] = shocks_coeffs[0:4]
    shocks[1, 1:] = shocks_coeffs[4:7]
    shocks[2, 2:] = shocks_coeffs[7:9]
    shocks[3, 3:] = shocks_coeffs[9:10]

    shocks_cov = shocks + shocks.T - np.diag(shocks.diagonal())

    # As we call the Cholesky decomposition, we need to handle the
    # special case of a deterministic model.
    if np.count_nonzero(shocks_cov) == 0:
        attr['optim_paras']['shocks_cholesky'] = np.zeros((4, 4))
    else:
        shocks_cholesky = np.linalg.cholesky(shocks_cov)
        attr['optim_paras']['shocks_cholesky'] = shocks_cholesky

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
    for optimizer in OPTIMIZERS:
        is_defined = (optimizer in ini.keys())
        if is_defined:
            attr['optimizer_options'][optimizer] = ini[optimizer]

    # We need to align the indicator for the fixed parameters.
    # In the initialization file, these refer to the upper triangular
    # matrix of the covariances. Inside the  program, we use the lower
    # triangular Cholesky decomposition.
    paras_fixed = attr['optim_paras']['paras_fixed'][:]

    paras_fixed_reordered = paras_fixed[:]

    for old, new in PARAS_MAPPING:
        paras_fixed_reordered[new] = paras_fixed[old]

    attr['optim_paras']['paras_fixed'] = paras_fixed_reordered

    return attr


def convert_attr_dict_to_init_dict(attr_dict):
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
    for old, new in PARAS_MAPPING:
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

    default = {
        'TYPE SHARES': {'coeffs': [], 'fixed': [], 'bounds': []},
        'TYPE SHIFTS': {'coeffs': [], 'fixed': [], 'bounds': []}
    }

    return default


def process_coefficient_line(line):
    """Extract the information about the estimation."""
    def _process_bounds(line):

        start, end = line.find('(') + 1, line.find(')')
        bounds = line[start:end].split(',')
        for i in range(2):
            if bounds[i].upper() == 'NONE':
                bounds[i] = None
            else:
                bounds[i] = float(bounds[i])

        return bounds

    list_ = shlex.split(line)

    # The coefficient is neither fixed nor are any bounds specified.
    if len(list_) == 2:
        is_fixed, bounds = False, [None, None]
    # The coefficient is both fixed and bounds are specified.
    elif len(list_) == 4:
        is_fixed, bounds = True, _process_bounds(line)
    # The coefficient is only fixed.
    elif (len(list_) == 3) and (list_[2] == '!'):
        is_fixed, bounds = True, [None, None]
    # The coefficient has only bounds
    elif len(list_) == 3 and (list_[2][0] == '('):
        is_fixed, bounds = False, _process_bounds(line)
    else:
        raise AssertionError

    return is_fixed, bounds


def _type_conversions(flag, value):
    """Convert the type of a value according to its flag."""
    # Type conversion
    if flag in ['agents', 'periods', 'start', 'max', 'draws', 'seed', 'points',
                'maxiter', 'maxfun', 'procs', 'npt', 'maxiter', 'm', 'maxls']:
        value = int(value)
    elif flag in ['file', 'options', 'type']:
        value = str(value)
    elif flag in ['debug', 'store', 'flag', 'mean']:
        assert (value.upper() in ['TRUE', 'FALSE'])
        value = (value.upper() == 'TRUE')
    elif flag in ['version', 'optimizer']:
        value = value.upper()
    else:
        value = float(value)

    # Finishing
    return value


def _add_to_dictionary(group, flag, value, is_fixed, bounds, dict_):
    """Process most parts of the initialization file."""
    # This aligns the label from the initialization file with the label inside
    # the RESPY logic.
    if flag in ['coeff', 'shift'] or (group == 'TYPES' and flag == 'share'):
        flag = 'coeffs'

    # Prepare container for information about coefficients
    if ('coeffs' not in dict_[group].keys()) and (flag in ['coeffs']):
        dict_[group]['coeffs'] = []
        dict_[group]['bounds'] = []
        dict_[group]['fixed'] = []
    elif group == 'EDUCATION' and flag not in dict_[group].keys():
        dict_[group][flag] = []

    # Collect information
    if flag in ['coeffs']:
        dict_[group]['coeffs'] += [value]
        dict_[group]['bounds'] += [bounds]
        dict_[group]['fixed'] += [is_fixed]
    elif group == 'EDUCATION' and (flag in ['share', 'start', 'lagged']):
        dict_[group][flag] += [value]
    elif flag not in ['shift'] or (group == 'TYPES' and flag == 'share'):
        dict_[group][flag] = value

    # Finishing.
    return dict_


def _determine_case(list_):
    """Process cases and determine whether group flag or empty line."""
    # Antibugging
    assert (isinstance(list_, list))

    # Get information
    is_empty = (len(list_) == 0)

    if not is_empty:
        is_group = list_[0].isupper()
        is_comment = list_[0] == '#'
    else:
        is_group = False
        is_comment = False

    # Antibugging
    assert is_group in [True, False]
    assert is_empty in [True, False]
    assert is_comment in [True, False]

    # Finishing
    return is_empty, is_group, is_comment
