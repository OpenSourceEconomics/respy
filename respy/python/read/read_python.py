import numpy as np
import shlex


def read(fname):
    """Read and process a RESPY initialization file."""
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

    ini_dict = default_model_dict()
    ini_dict.update(dict_)

    return ini_dict


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
