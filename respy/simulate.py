""" This module contains the interface to simulate a dataset and write the
information to disk.
"""

# standard library
import pandas as pd

import logging

# project library
from respy.python.simulate.simulate_auxiliary import start_logging
from respy.python.simulate.simulate_auxiliary import stop_logging
from respy.python.simulate.simulate_auxiliary import check_input
from respy.python.simulate.simulate_auxiliary import write_info
from respy.python.simulate.simulate_auxiliary import write_out

from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.shared.shared_auxiliary import get_respy_obj
from respy.python.shared.shared_auxiliary import check_dataset
from respy.python.shared.shared_auxiliary import create_draws

from respy.fortran.f2py_library import f2py_simulate

from respy.solve import solve

# Logging
from respy.python.simulate.simulate_python import pyth_simulate

logger = logging.getLogger('ROBUPY_SIMULATE')

''' Main function
'''


def simulate(input_, is_solved=False):
    """ Simulate dataset of synthetic agent following the model specified in
    the initialization file.
    """
    # Process input_
    respy_obj = get_respy_obj(input_)
    check_input(respy_obj, is_solved)

    # Solve the requested economy
    if not is_solved:
        solve(respy_obj)

    # Fire up the logging for the simulation. The logging of the solution
    # step is handled within the solution routines.
    start_logging()

    # Distribute class attributes
    periods_payoffs_systematic, mapping_state_idx, periods_emax, model_paras, \
        num_periods, num_agents_sim, states_all, edu_start, seed_sim, \
        is_debug, file_sim, edu_max, delta, version = \
            dist_class_attributes(respy_obj,
                'periods_payoffs_systematic', 'mapping_state_idx',
                'periods_emax', 'model_paras', 'num_periods', 'num_agents_sim',
                'states_all', 'edu_start', 'seed_sim', 'is_debug',
                'file_sim', 'edu_max', 'delta', 'version')

    # Auxiliary objects
    shocks_cholesky = dist_model_paras(model_paras, is_debug)[5]

    # Draw draws for the simulation.
    periods_draws_sims = create_draws(num_periods, num_agents_sim, seed_sim,
        is_debug)

    # Simulate a dataset with the results from the solution and write out the
    # dataset to a text file. In addition a file summarizing the dataset is
    # produced.
    logger.info('Starting simulation of model for ' + str(num_agents_sim) +
        ' agents with seed ' + str(seed_sim))

    # Collect arguments to pass in different implementations of the simulation.
    args = (periods_payoffs_systematic, mapping_state_idx, periods_emax,
        num_periods, states_all, num_agents_sim, edu_start, edu_max, delta,
        periods_draws_sims, shocks_cholesky)

    # Select appropriate interface
    if version == 'PYTHON':
        data_array = pyth_simulate(*args)
    elif version in ['FORTRAN', 'F2PY']:
        data_array = f2py_simulate(*args)
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

    stop_logging()

    # Finishing
    return data_frame, respy_obj


