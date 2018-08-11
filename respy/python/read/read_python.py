import numpy as np
import shlex
import os


def read(fname):
    """ Read and process a RESPY initialization file.
    """
    # Check input
    assert os.path.exists(fname)

    # Initialization
    dict_, group = {}, None

    with open(fname) as in_file:

        for line in in_file.readlines():

            # Split line
            list_ = shlex.split(line)

            # Determine special cases
            is_empty, is_group, is_comment = _process_cases(list_)

            # Applicability
            if is_empty or is_comment:
                continue

            # Prepare dictionary
            if is_group:
                group = list_[0]
                # Special treatment for OCCUPATION or TYPES, which consists of two entries.
                if group in ['OCCUPATION', 'TYPE']:
                    group = list_[0] + ' ' + list_[1]
                dict_[group] = {}
                continue

            # is the block for the optimization coefficients.
            flag, value = list_[:2]
            is_fixed, bounds = None, None

            if flag in ['coeff']:
                is_fixed, bounds = process_coefficient_line(line)

            # Type conversions
            value = _type_conversions(flag, value)

            # Process blocks of information
            dict_ = _process_line(group, flag, value, is_fixed, bounds, dict_)

    # For ease of usability, we allow to skip the TYPE SHARES and TYPE SHIFTS blocks. If they are
    # not present, then this implies that there is only a single type.
    for group in ['TYPE SHARES', 'TYPE SHIFTS']:
        if group in dict_.keys():
            continue

        dict_[group] = dict()
        for flag in ['coeffs', 'fixed', 'bounds']:
            dict_[group][flag] = []

    # Type conversion for SHOCKS
    dict_['SHOCKS']['coeffs'] = np.array(dict_['SHOCKS']['coeffs'])

    # Finishing
    return dict_


def process_coefficient_line(line):
    """ This function extracts the information about the estimation.
    """
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
    """ Type conversions
    """
    # Type conversion
    if flag in ['agents', 'periods', 'start', 'max', 'draws', 'seed', 'points', 'maxiter',
                'maxfun', 'procs', 'npt', 'maxiter', 'm', 'maxls', 'threads']:
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


def _process_line(group, flag, value, is_fixed, bounds, dict_):
    """ This function processes most parts of the initialization file.
    """
    # This aligns the label from the initialization file with the label inside the RESPY logic.
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


def _process_cases(list_):
    """ Process cases and determine whether group flag or empty line.
    """
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
