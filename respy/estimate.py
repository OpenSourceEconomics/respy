# project library
from respy.fortran.interface import resfort_interface
from respy.python.interface import respy_interface

from respy.python.estimate.estimate_auxiliary import check_input
from respy.python.process.process_python import process


def estimate(respy_obj):
    """ Estimate the model
    """
    # Read in estimation dataset. It only reads in the number of agents
    # requested for the estimation.
    data_frame = process(respy_obj)
    data_array = data_frame.as_matrix()

    # Antibugging
    assert check_input(respy_obj, data_frame)

    # Distribute class attributes
    version = respy_obj.get_attr('version')

    # Select appropriate interface
    if version in ['PYTHON']:
        x, val = respy_interface(respy_obj, 'estimate', data_array)
    elif version in ['FORTRAN']:
        x, val = resfort_interface(respy_obj, 'estimate', data_array)
    else:
        raise NotImplementedError

    # Finishing
    return x, val




