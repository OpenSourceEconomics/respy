""" This module contains the interface to the evaluation of the criterion
function.
"""

# standard library
import numpy as np

# project library
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
        raise NotImplementedError
    else:
        likl = evaluate_python(robupy_obj, data_frame)

    # Checks
    assert _check_evaluation(likl)

    # Finishing
    return likl


''' Auxiliary functions
'''


def _check_evaluation(likl):
    """ Check likelihood calculation.
    """

    assert isinstance(likl, float)
    assert np.isfinite(likl)

    # Finishing
    return True
