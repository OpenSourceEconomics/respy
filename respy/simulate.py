# standard library
import pandas as pd

import logging

# project library
from respy.solve import solve

from respy.python.simulate.simulate_auxiliary import logging_simulation
from respy.python.simulate.simulate_auxiliary import check_input
from respy.python.simulate.simulate_auxiliary import write_info
from respy.python.simulate.simulate_auxiliary import write_out

from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.shared.shared_auxiliary import check_dataset
from respy.python.shared.shared_auxiliary import create_draws

from respy.python.simulate.simulate_python import pyth_simulate

from respy.fortran.interface import resfort_interface
from respy.python.interface import respy_interface

logger = logging.getLogger('RESPY_SIMULATE')


def simulate(respy_obj):
    """ Simulate dataset of synthetic agent following the model specified in
    the initialization file.
    """

    version = respy_obj.get_attr('version')
    is_debug = respy_obj.get_attr('is_debug')

    # Select appropriate interface
    if version in ['PYTHON']:
        data_array = respy_interface(respy_obj, 'simulate')
    elif version in ['FORTRAN']:
        data_array = resfort_interface(respy_obj, 'simulate')
    else:
        raise NotImplementedError

    # Create pandas data frame with missing values.
    data_frame = pd.DataFrame(replace_missing_values(data_array))

    # Wrapping up by running some checks on the dataset and then writing out
    # the file and some basic information.
    if is_debug:
        check_dataset(data_frame, respy_obj, 'sim')

    write_out(respy_obj, data_frame)

    write_info(respy_obj, data_frame)

    logger.info('... finished \n')

    logging_simulation('stop')

    # Finishing
    return respy_obj


