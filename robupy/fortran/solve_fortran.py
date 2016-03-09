""" This module provides the interface to the functionality needed to solve the
model with FORTRAN.
"""
# standard library
import os

# module-wide variables
PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))

from robupy.fortran.auxiliary import _write_robufort_initialization
from robupy.fortran.auxiliary import _add_results

''' Main function
'''


def solve_fortran(robupy_obj):
    """ Solve dynamic programming using FORTRAN.
    """

    # Prepare ROBUFORT execution
    _write_robufort_initialization(robupy_obj, 'solve')

    # Call executable
    os.system('"' + PACKAGE_PATH + '/bin/robufort"')

    # Add results
    robupy_obj, _ = _add_results(robupy_obj, 'solve')

    # Finishing
    return robupy_obj

