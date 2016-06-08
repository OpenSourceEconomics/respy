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

from respy.fortran.fortran import resfort_interface

logger = logging.getLogger('RESPY_SIMULATE')


def simulate(respy_obj):
    """ Simulate dataset of synthetic agent following the model specified in
    the initialization file.
    """

    # Fire up the logging for the simulation. The logging of the solution
    # step is handled within the solution routines.
    logging_simulation('start')

    # Distribute class attributes
    model_paras, \
        num_periods, num_agents_sim, edu_start, seed_sim, \
        is_debug, edu_max, delta, version, model_paras, is_interpolated, \
        num_draws_emax, \
        num_points_interp, is_myopic, min_idx, seed_emax, num_agents_est, \
        num_draws_prob, tau, seed_prob, is_parallel, num_procs = \
            dist_class_attributes(respy_obj,
                'model_paras', 'num_periods', 'num_agents_sim',
                 'edu_start', 'seed_sim', 'is_debug',
                'edu_max', 'delta', 'version', 'model_paras',
                'is_interpolated', 'num_draws_emax', 'num_points_interp',
                'is_myopic', 'min_idx', 'seed_emax', 'num_agents_est',
                'num_draws_prob', 'tau', 'seed_prob', 'is_parallel', 'num_procs')

    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = \
        dist_model_paras(model_paras, is_debug)

    # Auxiliary objects
    shocks_cholesky = dist_model_paras(model_paras, is_debug)[-1]

    # Draw draws for the simulation.
    periods_draws_sims = create_draws(num_periods, num_agents_sim, seed_sim,
        is_debug)

    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax,
        is_debug)

    # Simulate a dataset with the results from the solution and write out the
    # dataset to a text file. In addition a file summarizing the dataset is
    # produced.
    logger.info('Starting simulation of model for ' + str(num_agents_sim) +
        ' agents with seed ' + str(seed_sim))

    # Collect arguments to pass in different implementations of the simulation.

    # Select appropriate interface
    if version == 'PYTHON':
        data_array = pyth_simulate(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cholesky, is_interpolated, num_draws_emax, num_periods,
            num_points_interp,
            is_myopic, edu_start, is_debug, edu_max, min_idx, delta,
            periods_draws_emax, num_agents_sim, periods_draws_sims)

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


