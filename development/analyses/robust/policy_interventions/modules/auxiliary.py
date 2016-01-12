""" This module contains some routines that help in the analysis of policy
responsiveness.
"""

# standard library
from robupy.clsRobupy import RobupyCls


def get_name(level, subsidy):
    """ Construct name from information about level and subsidy.
    """
    return '%03.3f' % level + '/' + '%.2f' % subsidy


def get_robupy_obj(init_dict):
    """ Get the object to pass in the solution method.
    """
    # Initialize and process class
    robupy_obj = RobupyCls()
    robupy_obj.set_attr('init_dict', init_dict)
    robupy_obj.lock()
    # Finishing
    return robupy_obj
