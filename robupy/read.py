""" This module contains the interface to read an initialization file from disk.
"""

# standard library
import shlex
import glob
import os

import numpy as np

# project library
from robupy.python.shared.shared_constants import ROOT_DIR
from robupy.python.read.clsRobupy import RobupyCls

''' Main function
'''


def read(fname, is_dict=False):
    """ Read and process a RESPY initialization file.
    """
    # Check input
    assert os.path.exists(fname)

    # Initialization
    dict_, keyword = {}, None

    with open(fname) as in_file:

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
            dict_ = _process_line(list_, dict_, keyword)

    # Type conversion for Shocks
    for key_ in ['coeffs', 'fixed']:
        dict_['SHOCKS'][key_] = np.array(dict_['SHOCKS'][key_])

    # Check quality.
    _check_integrity_read(dict_)

    # Construct container
    robupy_obj = RobupyCls(dict_)

    robupy_obj.lock()

    # Finishing.
    if is_dict:
        return robupy_obj, dict_
    else:
        return robupy_obj

''' Auxiliary functions
'''


def _process_line(list_, dict_, keyword):
    """ This function processes most parts of the initialization file.
    """
    # Distribute information
    name, val = list_[0], list_[1]

    # This aligns the label from the initialization file with the label
    # inside the ROBUPY logic.
    if name == 'coeff':
        name = 'coeffs'

    # Determine whether coefficient is fixed or not.
    is_fixed = None
    if name in ['coeffs']:
        is_fixed = (len(list_) == 3)
        if is_fixed:
            assert (list_[2] == '!')

    # Prepare container for information about coefficients
    if ('coeffs' not in dict_[keyword].keys()) and (name in ['coeffs']):
        dict_[keyword]['coeffs'] = []
        dict_[keyword]['fixed'] = []

    # Type conversion
    if name in ['agents', 'periods', 'start', 'max', 'draws',
        'seed', 'points', 'maxiter', 'maxfun']:
        val = int(val)
    elif name in ['file', 'options']:
        val = str(val)
    elif name in ['debug', 'store', 'apply']:
        assert (val.upper() in ['TRUE', 'FALSE'])
        val = (val.upper() == 'TRUE')
    elif name in ['version', 'optimizer']:
        val = val.upper()
    else:
        val = float(val)

    # Collect information
    if name in ['coeffs']:
        dict_[keyword]['coeffs'] += [val]
        dict_[keyword]['fixed'] += [is_fixed]
    else:
        dict_[keyword][name] = val

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

    # This allows to use the ROBUPY package also model initialization files
    # that are only valid for the RESPY package. Both will be separated at a
    # later stage.
    without_ambiguity = 'AMBIGUITY' not in dict_.keys()
    if without_ambiguity:
        dict_['AMBIGUITY'] = dict()
        dict_['AMBIGUITY']['level'] = 0.0

    # Check all required keys are defined. Note that additional keys might
    # exists due to the specification of optimizer options.
    keys_ = ['BASICS', 'EDUCATION', 'OCCUPATION A', 'OCCUPATION B']
    keys_ += ['HOME', 'INTERPOLATION', 'SHOCKS', 'SOLUTION']
    keys_ += ['AMBIGUITY', 'SIMULATION', 'PROGRAM', 'ESTIMATION']

    assert (set(keys_) <= set(dict_.keys()))

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
    assert (isinstance(dict_['SIMULATION']['agents'], int))
    assert (dict_['SIMULATION']['agents'] > 0)
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
        assert (np.all(np.isfinite(dict_['SHOCKS'][key_])))
        if key_ == 'coeffs':
            assert (dict_['SHOCKS'][key_].shape == (10,))
        elif key_ == 'fixed':
            assert (np.all(dict_['SHOCKS'][key_] == False)) or \
                (np.all(dict_['SHOCKS'][key_] == True))

    # Check AMBIGUITY
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
