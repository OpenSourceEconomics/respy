""" This module contains all the capabilities to solve the dynamic
programming problem.
"""

# standard library
import os

# project library
from robupy.auxiliary import write_robufort_initialization

# module-wide variables
PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))

''' Public function
'''


def solve_fortran(robupy_obj):
    """ Solve dynamic programming using FORTRAN.
    """
    # Distribute class attributes
    init_dict = robupy_obj.get_attr('init_dict')

    write_robufort_initialization(init_dict)

    os.system('"' + PACKAGE_PATH + '/bin/robufort"')

    # Finishing
    return  None
