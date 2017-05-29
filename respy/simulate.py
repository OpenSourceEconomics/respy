import pandas as pd
import os

from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.simulate.simulate_auxiliary import check_dataset_sim
from respy.python.shared.shared_constants import DATA_FORMATS_SIM
from respy.python.shared.shared_constants import DATA_LABELS_SIM
from respy.python.simulate.simulate_auxiliary import write_info
from respy.python.simulate.simulate_auxiliary import write_out
from respy.python.shared.shared_auxiliary import add_solution
from respy.fortran.interface import resfort_interface
from respy.python.interface import respy_interface


def simulate(respy_obj):
    """ Simulate dataset of synthetic agent following the model specified in the initialization
    file.
    """
    # Distribute class attributes
    is_debug, version, is_store, file_sim = dist_class_attributes(respy_obj, 'is_debug',
        'version', 'is_store', 'file_sim')

    # Cleanup
    for ext in ['sim', 'sol', 'amb', 'dat', 'info']:
        fname = file_sim + '.respy.' + ext
        if os.path.exists(fname):
            os.unlink(fname)

    # Select appropriate interface
    if version in ['PYTHON']:
        solution, data_array = respy_interface(respy_obj, 'simulate')
    elif version in ['FORTRAN']:
        solution, data_array = resfort_interface(respy_obj, 'simulate')
    else:
        raise NotImplementedError

    # Attach solution to class instance
    respy_obj = add_solution(respy_obj, *solution)

    respy_obj.unlock()
    respy_obj.set_attr('is_solved', True)
    respy_obj.lock()

    # Store object to file
    if is_store:
        respy_obj.store('solution.respy.pkl')

    # Create pandas data frame with missing values.
    data_frame = pd.DataFrame(replace_missing_values(data_array), columns=DATA_LABELS_SIM)
    data_frame = data_frame.astype(DATA_FORMATS_SIM)
    data_frame.set_index(['Identifier', 'Period'], drop=False, inplace=True)

    # Wrapping up by running some checks on the dataset and then writing out the file and some
    # basic information.
    if is_debug:
        check_dataset_sim(data_frame, respy_obj)

    write_out(respy_obj, data_frame)
    write_info(respy_obj, data_frame)

    # Finishing
    return respy_obj, data_frame


