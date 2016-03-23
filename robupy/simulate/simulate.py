""" This module contains the interface to simulate a dataset and write the
information to disk.

    Structure of Dataset:

        0   Identifier of Agent
        1   Time Period
        2   Choice (1 = Work A, 2 = Work B, 3 = Education, 4 = Home)
        3   Earnings (missing value if not working)
        4   Work Experience A
        5   Work Experience B
        6   Schooling
        7   Lagged Schooling

"""

# standard library
import logging

# project library
from robupy.simulate.simulate_auxiliary import check_simulation
from robupy.simulate.simulate_auxiliary import write_info
from robupy.simulate.simulate_auxiliary import write_out

from robupy.shared.auxiliary import distribute_class_attributes
from robupy.shared.auxiliary import check_dataset
from robupy.shared.auxiliary import create_draws

# Logging
from robupy.simulate.simulate_python import simulate_python

logger = logging.getLogger('ROBUPY_SIMULATE')

''' Main function
'''


def simulate(robupy_obj):
    """ Simulate dataset from model.
    """
    # Checks
    assert check_simulation(robupy_obj)

    # Distribute class attributes
    periods_payoffs_systematic, mapping_state_idx, periods_emax, model_paras, \
        num_periods, num_agents, states_all, edu_start, is_python, seed_data, \
        is_debug, file_sim, edu_max, delta = \
            distribute_class_attributes(robupy_obj,
                'periods_payoffs_systematic', 'mapping_state_idx',
                'periods_emax', 'model_paras', 'num_periods', 'num_agents',
                'states_all', 'edu_start', 'is_python', 'seed_data',
                'is_debug', 'file_sim', 'edu_max', 'delta')

    # Auxiliary objects
    shocks_cholesky = model_paras['shocks_cholesky']

    # Draw draws for the simulation.
    periods_draws_sims = create_draws(num_periods, num_agents, seed_data,
        is_debug, 'sims', shocks_cholesky)

    # Simulate a dataset with the results from the solution and write out the
    # dataset to a text file. In addition a file summarizing the dataset is
    # produced.
    logger.info('Staring simulation of model for ' + str(num_agents) +
        ' agents with seed ' + str(seed_data))

    # Simulate a dataset.
    args = (periods_payoffs_systematic, mapping_state_idx, periods_emax,
            num_periods, states_all, num_agents, edu_start, edu_max, delta,
            periods_draws_sims, is_python)

    data_frame = simulate_python(*args)

    # Wrapping up by running some checks on the dataset and then writing out
    # the file and some basic information.
    if is_debug:
        check_dataset(data_frame, robupy_obj)

    write_out(data_frame, file_sim)

    write_info(robupy_obj, data_frame)

    logger.info('... finished \n')

    # Finishing
    return data_frame


