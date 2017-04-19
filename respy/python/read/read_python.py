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
            is_empty, is_group = _process_cases(list_)

            # Applicability
            if is_empty:
                continue

            # Prepare dictionary
            if is_group:
                group = list_[0]
                # Special treatment for OCCUPATION, which consists of two
                # entries.
                if group == 'OCCUPATION':
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

    # Type conversion for Shocks
    dict_['SHOCKS']['coeffs'] = np.array(dict_['SHOCKS']['coeffs'])

    # TODO: Is this the right place to do it? No, but I will wait with the
    # refactoring until these are actual estimation parameters.
    if 'TYPES' not in dict_.keys():
        dict_['TYPES'] = dict()
        dict_['TYPES']['shares'] = [1.0]

    num_types = len(dict_['TYPES']['shares'])
    if num_types == 1:
        dict_['TYPES']['shifts'] = np.tile(0.0, (1, 4))
    else:
        dict_['TYPES']['shifts'] = np.reshape(dict_['TYPES']['shifts'], (num_types - 1, 4))
        dict_['TYPES']['shifts'] = np.concatenate((np.tile(0.0, (1, 4)),
                                                  dict_['TYPES']['shifts']), axis=0)

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
    if flag in ['agents', 'periods', 'start', 'max', 'draws',
        'seed', 'points', 'maxiter', 'maxfun', 'procs', 'npt', 'maxiter',
                'm', 'maxls']:
        value = int(value)
    elif flag in ['file', 'options', 'measure', 'type']:
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
    # This aligns the label from the initialization file with the label
    # inside the RESPY logic.
    if flag in ['coeff']:
        flag = 'coeffs'

    # Prepare container for information about coefficients
    if ('coeffs' not in dict_[group].keys()) and (flag in ['coeffs']):
        dict_[group]['coeffs'] = []
        dict_[group]['bounds'] = []
        dict_[group]['fixed'] = []

    # Collect information
    if flag in ['coeffs']:
        dict_[group]['coeffs'] += [value]
        dict_[group]['bounds'] += [bounds]
        dict_[group]['fixed'] += [is_fixed]
    elif flag not in ['share', 'shift']:
        dict_[group][flag] = value

    # Prepare container for information about types.
    if group == 'TYPES':
        if flag in ['share', 'shift']:
            flag += 's'
        if flag not in dict_[group].keys():
            dict_[group][flag] = []
        dict_[group][flag] += [value]

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
    else:
        is_group = False

    # Antibugging
    assert (is_group in [True, False])
    assert (is_empty in [True, False])

    # Finishing
    return is_empty, is_group
