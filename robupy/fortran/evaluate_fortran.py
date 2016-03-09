""" This module provides the interface to the functionality needed to
evaluate the likelihood function.
"""

# standard library
import os

# project library
from robupy.fortran.auxiliary import _write_robufort_initialization
from robupy.fortran.auxiliary import _add_results
from robupy.constants import HUGE_FLOAT

# module-wide variables
PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))

''' Main function
'''


def evaluate_fortran(robupy_obj, data_frame):
    """ Solve dynamic programming using FORTRAN.
    """
    # Prepare ROBUFORT execution
    _write_robufort_initialization(robupy_obj, 'evaluate')

    _write_dataset(data_frame)

    # Call executable
    os.system('"' + PACKAGE_PATH + '/bin/robufort"')

    # Add results
    robupy_obj, eval_ = _add_results(robupy_obj, 'evaluate')

    # Finishing
    return robupy_obj, eval_

''' Auxiliary function
'''


def _write_dataset(data_frame):
    """ Write the dataset to a temporary file. Missing values are set
    to large values.
    """
    with open('.data.robufort.dat', 'w') as file_:
        data_frame.to_string(file_, index=False,
            header=None, na_rep=str(HUGE_FLOAT))




















