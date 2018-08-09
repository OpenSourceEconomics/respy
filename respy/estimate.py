import atexit
import os

from respy.python.record.record_estimation import record_estimation_sample
from respy.python.shared.shared_auxiliary import remove_scratch
from respy.python.shared.shared_auxiliary import get_est_info

from respy.pre_processing.data_processing import process_dataset
from respy.fortran.interface import resfort_interface
from respy.python.interface import respy_interface


def estimate(respy_obj):
    """Estimate the model."""
    # Cleanup
    for fname in ['est.respy.log', 'est.respy.info']:
        if os.path.exists(fname):
            os.unlink(fname)

    if respy_obj.get_attr('is_solved'):
        respy_obj.reset()

    respy_obj.check_estimation()

    # This locks the estimation directory for additional estimation requests.
    atexit.register(remove_scratch, '.estimation.respy.scratch')
    open('.estimation.respy.scratch', 'w').close()

    # Read in estimation dataset. It only reads in the number of agents
    # requested for the estimation (or all available, depending on which is
    # less). It allows to read in only a subset of the initial conditions.
    data_frame = process_dataset(respy_obj)
    record_estimation_sample(data_frame)
    data_array = data_frame.as_matrix()

    # Distribute class attributes
    version = respy_obj.get_attr('version')

    # Select appropriate interface
    if version in ['PYTHON']:
        respy_interface(respy_obj, 'estimate', data_array)
    elif version in ['FORTRAN']:
        resfort_interface(respy_obj, 'estimate', data_array)
    else:
        raise NotImplementedError

    rslt = get_est_info()
    x, val = rslt['paras_step'], rslt['value_step']

    for fname in ['.estimation.respy.scratch', '.stop.respy.scratch']:
        remove_scratch(fname)

    # Finishing
    return x, val

