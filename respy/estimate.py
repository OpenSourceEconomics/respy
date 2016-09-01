import os

from respy.python.shared.shared_auxiliary import check_dataset
from respy.python.shared.shared_auxiliary import process_est_log
from respy.python.process.process_python import process
from respy.fortran.interface import resfort_interface
from respy.python.interface import respy_interface


def estimate(respy_obj):
    """ Estimate the model
    """
    # Read in estimation dataset. It only reads in the number of agents
    # requested for the estimation.
    data_frame = process(respy_obj)
    data_array = data_frame.as_matrix()

    # Cleanup
    for fname in ['est.respy.log', 'est.respy.info']:
        if os.path.exists(fname):
            os.unlink(fname)

    # Antibugging.
    assert _check_input(respy_obj, data_frame)

    # Distribute class attributes
    version = respy_obj.get_attr('version')

    # Select appropriate interface
    if version in ['PYTHON']:
        respy_interface(respy_obj, 'estimate', data_array)
    elif version in ['FORTRAN']:
        resfort_interface(respy_obj, 'estimate', data_array)
    else:
        raise NotImplementedError

    rslt = process_est_log()
    x, val = rslt['paras_final'], rslt['value_final']

    # Finishing
    return x, val


def _check_input(respy_obj, data_frame):
    """ Check input arguments.
    """
    # Check that class instance is locked.
    assert respy_obj.get_attr('is_locked')

    if respy_obj.get_attr('is_solved'):
        respy_obj.reset()

    # Check that dataset aligns with model specification.
    check_dataset(data_frame, respy_obj, 'est')

    # Finishing
    return respy_obj