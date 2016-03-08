""" This module contains the interface for the evaluation of the criterion
function.
"""

# standard library
import numpy as np

# project library
from robupy.fortran.evaluate_fortran import evaluate_fortran
from robupy.python.evaluate_python import evaluate_python
from robupy.auxiliary import check_dataset

''' Main function
'''


def evaluate(robupy_obj, data_frame):
    """ Evaluate likelihood function.
    """
    # Distribute class attribute
    is_debug = robupy_obj.get_attr('is_debug')

    version = robupy_obj.get_attr('version')

    # Check the dataset against the initialization files
    if is_debug:
        check_dataset(data_frame, robupy_obj)

    # Select appropriate interface
    if version == 'FORTRAN':
        robupy_obj, likl = evaluate_fortran(robupy_obj, data_frame)
    else:
        robupy_obj, likl = evaluate_python(robupy_obj, data_frame)

    # Checks
    assert _check_evaluation(likl)

    # Finishing
    return robupy_obj, likl

''' Auxiliary functions
'''


def _check_evaluation(likl):
    """ Check integrity of criterion function.
    """
    # Checks
    assert isinstance(likl, float)
    assert np.isfinite(likl)

    # Finishing
    return True
