""" This module contains the interface to read an initialization file from disk.
"""

# standard library
import numpy as np

import shlex
import sys
import os

# project library
from respy.python.read.read_auxiliary import check_line


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

            # Construct the required information for further processing.
            flag, value = list_[:2]
            is_fixed = None
            if flag in ['coeff']:
                is_fixed = (len(list_) == 3)
                if is_fixed:
                    assert (list_[2] == '!')

            # Type conversions
            value = _type_conversions(flag, value)

            # Distribute information
            check_line(group, flag, value)

            # Process blocks of information
            dict_ = _process_line(group, flag, value, is_fixed, dict_)

    # Type conversion for Shocks
    for key_ in ['coeffs', 'fixed']:
        dict_['SHOCKS'][key_] = np.array(dict_['SHOCKS'][key_])

    # Check SHOCKS. The covariance matrix can only be constructed after all
    # coefficients are processed. This then allow to check the basic
    # requirements.
    for key_ in ['coeffs', 'fixed']:
        assert (np.all(np.isfinite(dict_['SHOCKS'][key_])))
        if key_ == 'coeffs':
            assert (dict_['SHOCKS'][key_].shape == (10,))
        elif key_ == 'fixed':
            assert (np.all(dict_['SHOCKS'][key_] == False)) or \
                (np.all(dict_['SHOCKS'][key_] == True))

    # Check quality.
    _check_integrity_complete(dict_)

    # Finishing
    return dict_


def _type_conversions(flag, value):
    """ Type conversions
    """
    # Type conversion
    if flag in ['agents', 'periods', 'start', 'max', 'draws',
        'seed', 'points', 'maxiter', 'maxfun']:
        value = int(value)
    elif flag in ['file', 'options']:
        value = str(value)
    elif flag in ['debug', 'store', 'apply']:
        assert (value.upper() in ['TRUE', 'FALSE'])
        value = (value.upper() == 'TRUE')
    elif flag in ['version', 'optimizer']:
        value = value.upper()
    else:
        value = float(value)

    # Finishing
    return value


def _process_line(group, flag, value, is_fixed, dict_):
    """ This function processes most parts of the initialization file.
    """
    # This aligns the label from the initialization file with the label
    # inside the ROBUPY logic.
    if flag == 'coeff':
        flag = 'coeffs'

    # Prepare container for information about coefficients
    if ('coeffs' not in dict_[group].keys()) and (flag in ['coeffs']):
        dict_[group]['coeffs'] = []
        dict_[group]['fixed'] = []

    # Collect information
    if flag in ['coeffs']:
        dict_[group]['coeffs'] += [value]
        dict_[group]['fixed'] += [is_fixed]
    else:
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
    else:
        is_group = False

    # Antibugging
    assert (is_group in [True, False])
    assert (is_empty in [True, False])

    # Finishing
    return is_empty, is_group


def _check_integrity_complete(dict_):
    """ Perform some additional checks that are only possible after the full
    file is processed..
    """
    # Antibugging
    assert (isinstance(dict_, dict))

    try:
        for label in ['OCCUPATION A', 'OCCUPATION B']:
            assert (len(dict_[label]['coeffs']) == 6)
    except AssertionError:
        msg = '\n Too many coefficients in group OCCUPATION.\n'
        sys.exit(msg)

    try:
        assert (len(dict_['EDUCATION']['coeffs']) == 3)
    except AssertionError:
        msg = '\n Too many coefficients in group EDUCATION.\n'
        sys.exit(msg)

    try:
        assert (dict_['EDUCATION']['max'] > dict_['EDUCATION']['start'])
    except AssertionError:
        msg = '\n Maximum number of schooling less that start EDUCATION.\n'
        sys.exit(msg)

    try:
       assert (len(dict_['HOME']['coeffs']) == 1)
    except AssertionError:
        msg = '\n Too many coefficients in group HOME.\n'
        sys.exit(msg)

    # Check all required keys are defined. Note that additional keys might
    # exists due to the specification of optimizer options.
    keys_ = ['BASICS', 'EDUCATION', 'OCCUPATION A', 'OCCUPATION B']
    keys_ += ['HOME', 'INTERPOLATION', 'SHOCKS', 'SOLUTION']
    keys_ += ['SIMULATION', 'PROGRAM', 'ESTIMATION']
    try:
        assert (set(keys_) <= set(dict_.keys()))
    except AssertionError:
        msg = '\n Not all required groups specified.\n'
        sys.exit(msg)

    # Finishing
    return True
