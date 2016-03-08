
""" This module provides the interface to the functionality needed to solve the
model with FORTRAN.
"""
# standard library
import pandas as pd
import os

# module-wide variables
PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))

from robupy.fortran.auxiliary import _write_robufort_initialization
from robupy.fortran.auxiliary import _add_results

''' Main function
'''


def evaluate_fortran(robupy_obj, data_frame):
    """ Solve dynamic programming using FORTRAN.
    """

    # Prepare ROBUFORT execution
    _write_robufort_initialization(robupy_obj, 'evaluate')


    with open('.data.robufort.dat', 'w') as file_:

        data_frame.to_string(file_, index=False, header=None, na_rep='-99.0')


    # Call executable
    os.system('"' + PACKAGE_PATH + '/bin/robufort"')

    rslt = 0.0
    # Finishing
    return rslt


