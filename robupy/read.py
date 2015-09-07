""" This module contains all the capabilities for the processing and
simulating the model's initialization file.
"""

# standard library
import numpy as np
import shlex
import glob
import os

# project library
from robupy.clsRobupy import RobupyCls


''' Public function
'''


def read(file_):
    """ Process
    """
    # Initialization
    dict_ = {}

    for line in open(file_).readlines():

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
            dict_[keyword] = {}
            continue

        # Process blocks of information
        if keyword not in ['WORK', 'SHOCKS']:
            dict_ = _process_standard(list_, dict_, keyword)
        elif keyword in ['WORK']:
            _process_working(list_, dict_)
        elif keyword in ['SHOCKS']:
            dict_ = _process_shocks(list_, dict_)
        else:
            raise AssertionError

    # Cleanup dictionary
    del dict_['WORK']

    # Check quality.
    _check_integrity_read(dict_)

    # Construct container
    robupy_obj = RobupyCls()

    robupy_obj.set_attr('init_dict', dict_)

    robupy_obj.lock()

    # Finishing.
    return robupy_obj


''' Private functions
'''


def _process_shocks(list_, dict_):
    """ This function process the SHOCKS part of the initialization file.
    """
    # Distribute information
    if isinstance(dict_['SHOCKS'], dict):
        dict_['SHOCKS'] = []

    # Type conversion
    list_ = [float(i) for i in list_]

    # Collect information
    dict_['SHOCKS'] += [list_]

    # Finishing
    return dict_


def _process_working(list_, dict_):
    """ This function processes the WORKING part of the initialization file.
    """
    # Distribute information
    name, val_b, val_a = list_[0], list_[1], list_[2]

    # Initialize dictionary
    if 'A' not in dict_.keys():
        for subgroup in ['A', 'B']:
            dict_[subgroup] = {}
            dict_[subgroup]['coeff'] = []
            dict_[subgroup]['int'] = None

    # Type conversions
    val_b, val_a = float(val_b), float(val_a)

    # Collect information
    if name in ['coeff']:
        dict_['A'][name] += [val_b]
        dict_['B'][name] += [val_a]
    else:
        dict_['A'][name] = val_b
        dict_['B'][name] = val_a

    # Finishing
    return dict_


def _process_standard(list_, dict_, keyword):
    """ This function processes most parts of the initialization file.
    """
    # Distribute information
    name, val = list_[0], list_[1]

    # Prepare container.
    if (name not in dict_[keyword].keys()) and (name in ['coeff']):
        dict_[keyword][name] = []

    # Type conversion
    if name in ['agents', 'periods', 'start', 'max', 'draws', 'seed']:
        val = int(val)
    elif name in ['measure']:
        val = str(val)
    elif name in ['debug', 'fast']:
        assert (val.upper() in ['TRUE', 'FALSE'])
        val = (val.upper() == 'TRUE')
    else:
        val = float(val)

    # Collect information
    if name in ['coeff']:
        dict_[keyword][name] += [val]
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

    # Check all keys
    keys_ = ['BASICS', 'EDUCATION', 'A', 'B', 'HOME']
    keys_ += ['SHOCKS', 'SOLUTION', 'AMBIGUITY', 'SIMULATION']

    assert (set(keys_) == set(dict_.keys()))

    # Check BASICS
    assert (isinstance(dict_['BASICS']['periods'], int))
    assert (dict_['BASICS']['periods'] > 0)

    assert (isinstance(dict_['BASICS']['delta'], float))
    assert (dict_['BASICS']['delta'] >= 0)

    # Check WORK
    for label in ['A', 'B']:
        assert (isinstance(dict_[label]['int'], float))
        assert (np.isfinite(dict_[label]['int']))
        assert (len(dict_[label]['coeff']) == 5)
        assert (np.all(np.isfinite(dict_[label]['coeff'])))
        assert (all(isinstance(coeff, float) for coeff in dict_[label][
            'coeff']))

    # Check EDUCATION
    for label in ['start', 'max']:
        assert (isinstance(dict_['EDUCATION'][label], int))
        assert (dict_['EDUCATION'][label] >= 0)

    assert (dict_['EDUCATION']['max'] > dict_['EDUCATION']['start'])

    assert (isinstance(dict_['EDUCATION']['int'], float))
    assert (np.isfinite(dict_['EDUCATION']['int']))

    assert (len(dict_['EDUCATION']['coeff']) == 2)
    assert (np.all(np.isfinite(dict_['EDUCATION']['coeff'])))
    assert (all(isinstance(coeff, float) for coeff in dict_['EDUCATION'][
            'coeff']))

    # Check HOME
    assert (isinstance(dict_['HOME']['int'], float))
    assert (np.isfinite(dict_['HOME']['int']))

    # Check SOLUTION
    assert (isinstance(dict_['SOLUTION']['draws'], int))
    assert (dict_['SOLUTION']['draws'] >= 0)
    assert (isinstance(dict_['SOLUTION']['seed'], int))
    assert (dict_['SOLUTION']['seed'] >= 0)
    assert (dict_['SOLUTION']['debug'] in [True, False])
    assert (dict_['SOLUTION']['fast'] in [True, False])

    if dict_['SOLUTION']['fast'] is True:
        package_dir = os.path.dirname(os.path.realpath(__file__))
        assert (len(glob.glob(package_dir +
                              '/python/f2py/f2py_core.*.so')) == 1)

    # Check SHOCKS
    assert (len(dict_['SHOCKS']) == 4)
    assert (np.array(dict_['SHOCKS']).shape == (4, 4))
    assert (np.all(np.isfinite(np.array(dict_['SHOCKS']))))
    assert (np.all(np.diag(np.array(dict_['SHOCKS']) >= 0)))
    assert ((np.array(dict_['SHOCKS']).transpose() ==
             np.array(dict_['SHOCKS'])).all())
    if not (np.count_nonzero(dict_['SHOCKS']) == 0):
        assert(np.linalg.det(dict_['SHOCKS']) > 0)

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

    # Temporary restrictions
    assert (dict_['AMBIGUITY']['level'] >= 0.00)
    assert (dict_['AMBIGUITY']['measure'] in ['absolute', 'kl'])

    # Finishing
    return True
