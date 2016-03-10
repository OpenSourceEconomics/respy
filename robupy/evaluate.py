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
    shocks_zero = robupy_obj.get_attr('shocks_zero')

    version = robupy_obj.get_attr('version')

    # Check the dataset against the initialization files
    assert _check_evaluation('in', data_frame, robupy_obj, shocks_zero)

    # Select appropriate interface
    if version == 'FORTRAN':
        robupy_obj, likl = evaluate_fortran(robupy_obj, data_frame)
    else:
        robupy_obj, likl = evaluate_python(robupy_obj, data_frame)

    # Checks
    assert _check_evaluation('out', likl)

    # Finishing
    return robupy_obj, likl

''' Auxiliary functions
'''


def _check_evaluation(str_, *args):
    """ Check integrity of criterion function.
    """
    if str_ == 'out':

        # Distribute input parameters
        likl, = args

        # Check quality
        assert isinstance(likl, float)
        assert np.isfinite(likl)

    elif str_ == 'in':

        # Distribute input parameters
        data_frame, robupy_obj, shocks_zero = args

        # Check quality
        check_dataset(data_frame, robupy_obj)
        assert (shocks_zero is False)

    # Finishing
    return True
