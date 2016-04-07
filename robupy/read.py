""" This module contains the interface to read an initialization file from disk.
"""

# standard library
import glob
import os
import shlex

import numpy as np

# project library
from robupy.python.shared.shared_constants import ROOT_DIR
from robupy.python.read.clsRobupy import RobupyCls

''' Main function
'''


def read(file_):
    """ Read an initialization file from disk.
    """
    # Initialization
    dict_ = {}

    with open(file_) as in_file:

        for line in in_file.readlines():

            # Split line
            list_ = shlex.split(line)

            # Determine special cases
            is_empty, is_keyword = _process_cases(list_)

            # Applicability
            if is_empty:
                continue

            # Prepare dictionary
            if is_keyword:
                keyword = list_[0]
                # Special treatment for OCCUPATION, which consists of two
                # entries.
                if keyword == 'OCCUPATION':
                    keyword = list_[0] + ' ' + list_[1]
                dict_[keyword] = {}
                continue

            # Process blocks of information
            if keyword in ['SHOCKS']:
                dict_ = _process_shocks(list_, dict_)
            else:
                dict_ = _process_standard(list_, dict_, keyword)

    # Type conversion for Shocks
    for key_ in ['coeffs', 'fixed']:
        dict_['SHOCKS'][key_] = np.array(dict_['SHOCKS'][key_])

    # Check quality.
    _check_integrity_read(dict_)

    # Construct container
    robupy_obj = RobupyCls(dict_)

    robupy_obj.lock()

    # Finishing.
    return robupy_obj

''' Auxiliary functions
'''


def _process_shocks(list_, dict_):
    """ This function process the SHOCKS part of the initialization file.
    """
    # Distribute information
    if not dict_['SHOCKS'].keys():
        dict_['SHOCKS'] = {}
        dict_['SHOCKS']['coeffs'] = []
        dict_['SHOCKS']['fixed'] = []

    # Type conversion
    values = []
    fixed_indicators = []
    for i, val in enumerate(list_):

        is_fixed = val[0] == '!'
        if is_fixed:
            val = val[1:]

        values += [float(val)]
        fixed_indicators += [is_fixed]

    # Collect information
    dict_['SHOCKS']['coeffs'] += [values]
    dict_['SHOCKS']['fixed'] += [fixed_indicators]

    # Finishing
    return dict_


def _process_standard(list_, dict_, keyword):
    """ This function processes most parts of the initialization file.
    """
    # Distribute information
    name, val = list_[0], list_[1]

    # This aligns the label from the initialization file with the label
    # inside the ROBUPY logic.
    if name == 'coeff':
        name = 'coeffs'

    # Determine whether coefficient is fixed or not.
    is_fixed = val[0] == '!'
    if is_fixed:
        val = val[1:]

    # Prepare container.
    if ('coeffs' not in dict_[keyword].keys()) and (name in ['coeffs', 'int']):
        dict_[keyword]['coeffs'] = []
        dict_[keyword]['fixed'] = []

    # Type conversion
    if name in ['agents', 'periods', 'start', 'max', 'draws',
        'seed', 'points', 'maxiter']:
        val = int(val)
    elif name in ['measure', 'file']:
        val = str(val)
    elif name in ['debug', 'store', 'apply']:
        assert (val.upper() in ['TRUE', 'FALSE'])
        val = (val.upper() == 'TRUE')
    elif name in ['version', 'optimizer']:
        val = val.upper()
    else:
        val = float(val)

    # Collect information
    if name in ['coeffs', 'int']:
        dict_[keyword]['coeffs'] += [val]
        dict_[keyword]['fixed'] += [is_fixed]
    else:
        dict_[keyword][name] = val

    # Move value of intercept to first position.
    if name == 'int':
        dict_[keyword]['coeffs'].insert(0, dict_[keyword]['coeffs'][-1])
        dict_[keyword]['coeffs'].pop()

        dict_[keyword]['fixed'].insert(0, dict_[keyword]['fixed'][-1])
        dict_[keyword]['fixed'].pop()

    # Finishing.
    return dict_


def _process_cases(list_):
    """ Process cases and determine whether keyword or empty line.
    """
    # Antibugging
    assert (isinstance(list_, list))

    # Get information
    is_empty = (len(list_) == 0)

    if not is_empty:
        is_keyword = list_[0].isupper()
    else:
        is_keyword = False

    # Antibugging
    assert (is_keyword in [True, False])
    assert (is_empty in [True, False])

    # Finishing
    return is_empty, is_keyword


def _check_integrity_read(dict_):
    """ Check integrity of initialization dictionary.
    """
    # Antibugging
    assert (isinstance(dict_, dict))

    # Check all keys
    keys_ = ['BASICS', 'EDUCATION', 'OCCUPATION A', 'OCCUPATION B']
    keys_ += ['HOME', 'INTERPOLATION']
    keys_ += ['SHOCKS', 'SOLUTION', 'AMBIGUITY', 'SIMULATION', 'PROGRAM']
    keys_ += ['ESTIMATION']

    assert (set(keys_) == set(dict_.keys()))

    # Check BASICS
    assert (isinstance(dict_['BASICS']['periods'], int))
    assert (dict_['BASICS']['periods'] > 0)

    assert (isinstance(dict_['BASICS']['delta'], float))
    assert (dict_['BASICS']['delta'] >= 0)

    # Check OCCUPATIONS
    for label in ['OCCUPATION A', 'OCCUPATION B']:
        assert (len(dict_[label]['coeffs']) == 6)
        assert (np.all(np.isfinite(dict_[label]['coeffs'])))
        assert (all(isinstance(coeff, float) for coeff in dict_[label][
            'coeffs']))

    # Check EDUCATION
    for label in ['start', 'max']:
        assert (isinstance(dict_['EDUCATION'][label], int))
        assert (dict_['EDUCATION'][label] >= 0)

    assert (dict_['EDUCATION']['max'] > dict_['EDUCATION']['start'])

    assert (len(dict_['EDUCATION']['coeffs']) == 3)
    assert (np.all(np.isfinite(dict_['EDUCATION']['coeffs'])))
    assert (all(isinstance(coeff, float) for coeff in dict_['EDUCATION'][
            'coeffs']))

    # Check HOME
    assert (len(dict_['HOME']['coeffs']) == 1)
    assert (np.isfinite(dict_['HOME']['coeffs'][0]))
    assert (isinstance(dict_['HOME']['coeffs'][0], float))

    # Check ESTIMATION
    assert (isinstance(dict_['ESTIMATION']['draws'], int))
    assert (dict_['ESTIMATION']['draws'] >= 0)
    assert (isinstance(dict_['ESTIMATION']['seed'], int))
    assert (dict_['ESTIMATION']['seed'] >= 0)
    assert (isinstance(dict_['ESTIMATION']['file'], str))
    assert (isinstance(dict_['ESTIMATION']['tau'], float))
    assert (dict_['ESTIMATION']['tau'] >= 0)

    # Check SOLUTION
    assert (isinstance(dict_['SOLUTION']['draws'], int))
    assert (dict_['SOLUTION']['draws'] >= 0)
    assert (isinstance(dict_['SOLUTION']['seed'], int))
    assert (dict_['SOLUTION']['seed'] >= 0)
    assert (dict_['SOLUTION']['store'] in [True, False])

    # Check PROGRAM
    assert (dict_['PROGRAM']['debug'] in [True, False])
    assert (dict_['PROGRAM']['version'] in ['FORTRAN', 'F2PY', 'PYTHON'])

    if dict_['PROGRAM']['version'] == 'F2PY':
        assert (len(glob.glob(ROOT_DIR + '/fortran/f2py_library.*.so')) == 1)

    if dict_['PROGRAM']['version'] == 'FORTRAN':
        assert (os.path.exists(ROOT_DIR + '/fortran/bin/robufort'))

    # Check SHOCKS
    for key_ in ['coeffs', 'fixed']:
        assert dict_['SHOCKS'][key_].shape == (4, 4)
        assert (np.all(np.isfinite(dict_['SHOCKS'][key_])))
        if key_ == 'coeffs':
            assert (np.all(np.diag(dict_['SHOCKS'][key_]) >= 0))
            assert (np.array_equal(dict_['SHOCKS'][key_].transpose(),
                dict_['SHOCKS'][key_]))
            if not (np.count_nonzero(dict_['SHOCKS'][key_]) == 0):
                assert(np.linalg.det(dict_['SHOCKS'][key_]) > 0)
        elif key_ == 'fixed':
            assert (np.all(dict_['SHOCKS'][key_] == False)) or \
                (np.all(dict_['SHOCKS'][key_] == True))

    # Check AMBIGUITY
    assert (dict_['AMBIGUITY']['measure'] in ['kl', 'absolute'])
    assert (isinstance(dict_['AMBIGUITY']['level'], float))
    assert (dict_['AMBIGUITY']['level'] >= 0.00)
    assert (np.isfinite(dict_['AMBIGUITY']['level']))

    # Check SIMULATION
    assert (isinstance(dict_['SIMULATION']['agents'], int))
    assert (dict_['SIMULATION']['agents'] > 0)
    assert (isinstance(dict_['SIMULATION']['seed'], int))
    assert (dict_['SIMULATION']['seed'] >= 0)
    assert (isinstance(dict_['SIMULATION']['file'], str))

    # Check INTERPOLATION
    assert (dict_['INTERPOLATION']['apply'] in [True, False])
    assert (isinstance(dict_['INTERPOLATION']['points'], int))
    assert (dict_['INTERPOLATION']['points'] > 0)

    # Finishing
    return True
