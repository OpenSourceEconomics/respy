""" This module contains all the capabilities for the processing and
simulating the model's initialization file.
"""

# standard library
import numpy as np
import shlex

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
    _check_integrity_process(dict_)

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
    if name not in dict_[keyword].keys():
        if name in ['coeff']:
            dict_[keyword][name] = []

    # Type conversion
    if name in ['agents', 'periods', 'initial', 'maximum', 'draws', 'seed']:
        val = int(val)
    elif name in []:
        val = str(val)
    elif name in ['debug']:
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


def _check_integrity_process(dict_):
    """ Check integrity of initialization dictionary.
    """
    # Antibugging
    assert (isinstance(dict_, dict))

    # Check all keys
    keys_ = ['BASICS', 'EDUCATION', 'A', 'B', 'HOME']
    keys_ += ['SHOCKS', 'COMPUTATION']

    assert (set(keys_) == set(dict_.keys()))

    # Check BASICS
    assert (isinstance(dict_['BASICS']['agents'], int))
    assert (dict_['BASICS']['agents'] > 0)

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
    for label in ['initial', 'maximum']:
        assert (isinstance(dict_['EDUCATION'][label], int))
        assert (dict_['EDUCATION'][label] >= 0)

    assert (dict_['EDUCATION']['maximum'] > dict_['EDUCATION']['initial'])

    assert (isinstance(dict_['EDUCATION']['int'], float))
    assert (np.isfinite(dict_['EDUCATION']['int']))

    assert (len(dict_['EDUCATION']['coeff']) == 2)
    assert (np.all(np.isfinite(dict_['EDUCATION']['coeff'])))
    assert (all(isinstance(coeff, float) for coeff in dict_['EDUCATION'][
            'coeff']))

    # Check HOME
    assert (isinstance(dict_['HOME']['int'], float))
    assert (np.isfinite(dict_['HOME']['int']))

    # Check COMPUTATION
    assert (isinstance(dict_['COMPUTATION']['draws'], int))
    assert (dict_['COMPUTATION']['draws'] >= 0)
    assert (isinstance(dict_['COMPUTATION']['seed'], int))
    assert (dict_['COMPUTATION']['seed'] >= 0)

    # Check SHOCKS
    assert (len(dict_['SHOCKS']) == 4)
    assert (np.array(dict_['SHOCKS']).shape == (4, 4))
    assert (np.all(np.isfinite(np.array(dict_['SHOCKS']))))
    assert (np.all(np.diag(np.array(dict_['SHOCKS']) > 0)))
    assert ((np.array(dict_['SHOCKS']).transpose() ==
             np.array(dict_['SHOCKS'])).all())

    # Finishing
    return True
