""" This module serves as the interface to the basic PYTHON functionality.
"""

# standard library
import logging

# project library
from respy.python.estimate.estimate_auxiliary import get_optim_paras
from respy.python.estimate.estimate_wrapper import OptimizationClass

from respy.python.simulate.simulate_python import pyth_simulate

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.shared.shared_auxiliary import create_draws

from respy.python.solve.solve_python import pyth_solve

logger = logging.getLogger('RESPY_SIMULATE')


def respy_interface(respy_obj, request, data_array=None):

    # Distribute class attributes
    model_paras, num_periods, num_agents_est, edu_start, is_debug, edu_max, \
        delta, num_draws_prob, seed_prob, num_draws_emax, seed_emax, \
        min_idx, is_myopic, is_interpolated, num_points_interp, version, \
        maxiter, optimizer_used, tau, paras_fixed, optimizer_options, \
        is_parallel, num_procs, seed_sim, num_agents_sim = \
            dist_class_attributes( respy_obj, 'model_paras', 'num_periods',
                'num_agents_est', 'edu_start', 'is_debug', 'edu_max', 'delta',
                'num_draws_prob', 'seed_prob', 'num_draws_emax', 'seed_emax',
                'min_idx', 'is_myopic', 'is_interpolated',
                'num_points_interp', 'version', 'maxiter', 'optimizer_used',
                'tau', 'paras_fixed', 'optimizer_options', 'is_parallel',
                'num_procs', 'seed_sim', 'num_agents_sim')

    # Auxiliary objects
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = dist_model_paras(
        model_paras, is_debug)

    if request == 'estimate':

        periods_draws_prob = create_draws(num_periods, num_draws_prob,
            seed_prob, is_debug)

        # Draw standard normal deviates for the solution and evaluation step.
        periods_draws_emax = create_draws(num_periods, num_draws_emax,
            seed_emax, is_debug)

        # Construct starting values
        x_free_start = get_optim_paras(coeffs_a, coeffs_b, coeffs_edu,
            coeffs_home, shocks_cholesky, 'free', paras_fixed, is_debug)

        x_all_start = get_optim_paras(coeffs_a, coeffs_b, coeffs_edu,
            coeffs_home, shocks_cholesky, 'all', paras_fixed, is_debug)

        # Collect arguments that are required for the criterion function. These
        # must be in the correct order already.
        args = (is_interpolated, num_draws_emax, num_periods, num_points_interp,
            is_myopic, edu_start, is_debug, edu_max, min_idx, delta, data_array,
            num_agents_est, num_draws_prob, tau, periods_draws_emax,
            periods_draws_prob)

        opt_obj = OptimizationClass()

        opt_obj.set_attr('args', args)

        opt_obj.set_attr('optimizer_options', optimizer_options)

        opt_obj.set_attr('x_info', (x_all_start, paras_fixed))

        opt_obj.set_attr('optimizer_used', optimizer_used)

        opt_obj.set_attr('version', version)

        opt_obj.set_attr('maxiter', maxiter)

        opt_obj.lock()

        # Perform optimization.
        args = opt_obj.optimize(x_free_start)

    elif request == 'simulate':

        # Draw draws for the simulation.
        periods_draws_sims = create_draws(num_periods, num_agents_sim, seed_sim,
            is_debug)

        # Draw standard normal deviates for the solution and evaluation step.
        periods_draws_emax = create_draws(num_periods, num_draws_emax,
            seed_emax, is_debug)

        # Simulate a dataset with the results from the solution and write out
        # the dataset to a text file. In addition a file summarizing the
        # dataset is produced.
        logger.info('Starting simulation of model for ' + str(
            num_agents_sim) + ' agents with seed ' + str(seed_sim))

        # Collect arguments to pass in different implementations of the
        # simulation.
        data_array = pyth_simulate(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cholesky, is_interpolated, num_draws_emax, num_periods,
            num_points_interp, is_myopic, edu_start, is_debug, edu_max, min_idx,
            delta, periods_draws_emax, num_agents_sim, periods_draws_sims)

        args = data_array

    elif request == 'solve':

        # Draw standard normal deviates for the solution and evaluation step.
        periods_draws_emax = create_draws(num_periods, num_draws_emax,
            seed_emax, is_debug)

        # Collect baseline arguments. These are latter amended to account for
        # each interface.
        args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
            is_interpolated, num_draws_emax, num_periods, num_points_interp,
            is_myopic, edu_start, is_debug, edu_max, min_idx, delta, periods_draws_emax)

        args = pyth_solve(*args)

    return args


