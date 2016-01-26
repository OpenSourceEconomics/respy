""" Some auxiliary functions.
"""

# standard library
import pickle as pkl

import glob
import sys
import os

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR)

# robupy library
from robupy.clsRobupy import RobupyCls


def get_float_directories():
    """ Get directories that have a float-type name.
    """
    # Get all possible files.
    candidates = glob.glob('*')
    directories = []
    for candidate in candidates:
        # Check if directory at all.
        if not os.path.isdir(candidate):
            continue
        # Check if directory with float-type name.
        try:
            float(candidate)
        except ValueError:
            continue
        # Collect survivors.
        directories += [float(candidate)]
    # Finishing
    return directories


def float_to_string(float_):
    """ Get string from a float.
    """
    return '%03.4f' % float_


def get_robupy_obj(init_dict):
    """ Get the object to pass in the solution method.
    """
    # Initialize and process class
    robupy_obj = RobupyCls()
    robupy_obj.set_attr('init_dict', init_dict)
    robupy_obj.lock()
    # Finishing
    return robupy_obj


def check_indifference():
    """ Check whether the results for the indifference points are available.
    """
    assert os.path.exists('../indifference_curve/rslts/indifference_curve'
                          '.robupy''.pkl')


def get_indifference_points():
    """ Get the points of indifference.
    """
    pairs = pkl.load(open('../indifference_curve/rslts/indifference_curve'
                              '.robupy.pkl', 'rb'))['opt']

    # Reorganize the tuples.
    specifications = []
    for level in pairs.keys():
        specifications += [(level, pairs[level])]

    # Finishing
    return specifications

