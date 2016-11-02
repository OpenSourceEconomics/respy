from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import approx_fprime
from scipy.optimize import fmin_powell
from scipy.optimize import fmin_bfgs

import numpy as np

from respy.python.record.record_estimation import record_estimation_scalability
from respy.python.record.record_estimation import record_estimation_scaling
from respy.python.record.record_estimation import record_estimation_final
from respy.python.record.record_estimation import record_estimation_stop
from respy.python.solve.solve_auxiliary import pyth_create_state_space
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.estimate.estimate_wrapper import OptimizationClass
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.simulate.simulate_python import pyth_simulate
from respy.python.estimate.estimate_wrapper import MaxfunError
from respy.python.shared.shared_auxiliary import apply_scaling
from respy.python.shared.shared_auxiliary import create_draws
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.python.solve.solve_python import pyth_solve


def respy_interface(respy_obj, request, data_array=None):
    """ This function provides the interface to the PYTHOn functionality.
    """
    # Distribute class attributes
    model_paras, num_periods, edu_start, is_debug, edu_max, \
        delta, num_draws_prob, seed_prob, num_draws_emax, seed_emax, \
        min_idx, is_myopic, is_interpolated, num_points_interp, maxfun, \
        optimizer_used, tau, paras_fixed, optimizer_options, seed_sim, \
        num_agents_sim, measure, file_sim, paras_bounds, preconditioning = \
            dist_class_attributes(respy_obj, 'model_paras', 'num_periods',
                'edu_start', 'is_debug', 'edu_max', 'delta', 'num_draws_prob',
                'seed_prob', 'num_draws_emax', 'seed_emax', 'min_idx',
                'is_myopic', 'is_interpolated', 'num_points_interp', 'maxfun',
                'optimizer_used', 'tau', 'paras_fixed', 'optimizer_options',
                'seed_sim', 'num_agents_sim', 'measure', 'file_sim',
                'paras_bounds', 'preconditioning')

    level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = \
        dist_model_paras(model_paras, is_debug)

    if request == 'estimate':

        periods_draws_prob = create_draws(num_periods, num_draws_prob,
            seed_prob, is_debug)

        # Draw standard normal deviates for the solution and evaluation step.
        periods_draws_emax = create_draws(num_periods, num_draws_emax,
            seed_emax, is_debug)

        # Construct starting values
        x_optim_free_unscaled_start = get_optim_paras(level, coeffs_a,
            coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, 'free',
            paras_fixed, is_debug)

        x_optim_all_unscaled_start = get_optim_paras(level, coeffs_a, coeffs_b,
            coeffs_edu, coeffs_home, shocks_cholesky, 'all', paras_fixed,
            is_debug)

        # Construct the state space
        states_all, states_number_period, mapping_state_idx, \
            max_states_period = pyth_create_state_space(num_periods,
                edu_start, edu_max, min_idx)

        # Cutting to size
        states_all = states_all[:, :max(states_number_period), :]

        # Collect arguments that are required for the criterion function. These
        # must be in the correct order already.
        args = (is_interpolated, num_draws_emax, num_periods,
            num_points_interp, is_myopic, edu_start, is_debug, edu_max, delta,
            data_array, num_draws_prob, tau, periods_draws_emax,
            periods_draws_prob, states_all, states_number_period,
            mapping_state_idx, max_states_period, measure, optimizer_options)

        # Special case where just an evaluation at the starting values is
        # requested is accounted for. Note, that the relevant value of the
        # criterion function is always the one indicated by the class
        # attribute and not the value returned by the optimization algorithm.

        num_free = paras_fixed.count(False)

        paras_bounds_free_unscaled = []
        for i in range(27):
            if not paras_fixed[i]:
                lower, upper = paras_bounds[i][:]
                if lower is None:
                    lower = -HUGE_FLOAT
                else:
                    lower = lower

                if upper is None:
                    upper = HUGE_FLOAT
                else:
                    upper = upper

                paras_bounds_free_unscaled += [[lower, upper]]

        paras_bounds_free_unscaled = np.array(paras_bounds_free_unscaled)

        precond_matrix = get_precondition_matrix(preconditioning, paras_fixed,
            x_optim_all_unscaled_start, args, maxfun)

        x_optim_free_scaled_start = apply_scaling(x_optim_free_unscaled_start,
            precond_matrix, 'do')

        paras_bounds_free_scaled = np.tile(np.nan, (num_free, 2))
        for i in range(2):
            paras_bounds_free_scaled[:, i] = apply_scaling(
                paras_bounds_free_unscaled[:, i], precond_matrix, 'do')

        record_estimation_scaling(x_optim_free_unscaled_start,
            x_optim_free_scaled_start, paras_bounds_free_scaled,
            precond_matrix, paras_fixed)

        opt_obj = OptimizationClass()
        opt_obj.maxfun = maxfun
        opt_obj.paras_fixed = paras_fixed
        opt_obj.x_optim_all_unscaled_start = x_optim_all_unscaled_start
        opt_obj.precond_matrix = precond_matrix

        if maxfun == 0:

            record_estimation_scalability('Start')
            opt_obj.crit_func(x_optim_free_scaled_start, *args)
            record_estimation_scalability('Finish')

            success = True
            message = 'Single evaluation of criterion function at starting ' \
                      'values.'

        elif optimizer_used == 'SCIPY-BFGS':

            bfgs_maxiter = optimizer_options['SCIPY-BFGS']['maxiter']
            bfgs_gtol = optimizer_options['SCIPY-BFGS']['gtol']
            bfgs_eps = optimizer_options['SCIPY-BFGS']['eps']

            try:
                rslt = fmin_bfgs(opt_obj.crit_func, x_optim_free_scaled_start,
                    args=args, gtol=bfgs_gtol, epsilon=bfgs_eps,
                    maxiter=bfgs_maxiter, full_output=True, disp=False)

                success = (rslt[6] not in [1, 2])
                message = 'Optimization terminated successfully.'
                if rslt[6] == 1:
                    message = 'Maximum number of iterations exceeded.'
                elif rslt[6] == 2:
                    message = 'Gradient and/or function calls not changing.'

            except MaxfunError:
                success = False
                message = 'Maximum number of iterations exceeded.'

        elif optimizer_used == 'SCIPY-LBFGSB':

            lbfgsb_maxiter = optimizer_options['SCIPY-LBFGSB']['maxiter']
            lbfgsb_maxls = optimizer_options['SCIPY-LBFGSB']['maxls']
            lbfgsb_factr = optimizer_options['SCIPY-LBFGSB']['factr']
            lbfgsb_pgtol = optimizer_options['SCIPY-LBFGSB']['pgtol']
            lbfgsb_eps = optimizer_options['SCIPY-LBFGSB']['eps']
            lbfgsb_m = optimizer_options['SCIPY-LBFGSB']['m']

            try:
                rslt = fmin_l_bfgs_b(opt_obj.crit_func,
                    x_optim_free_scaled_start, args=args, approx_grad=True,
                    bounds=paras_bounds_free_scaled, m=lbfgsb_m,
                    factr=lbfgsb_factr, pgtol=lbfgsb_pgtol,
                    epsilon=lbfgsb_eps, iprint=-1, maxfun=maxfun,
                    maxiter=lbfgsb_maxiter, maxls=lbfgsb_maxls)

                success = (rslt[2]['warnflag'] in [0])
                message = rslt[2]['task']

            except MaxfunError:
                success = False
                message = 'Maximum number of iterations exceeded.'

        elif optimizer_used == 'SCIPY-POWELL':

            powell_maxiter = optimizer_options['SCIPY-POWELL']['maxiter']
            powell_maxfun = optimizer_options['SCIPY-POWELL']['maxfun']
            powell_xtol = optimizer_options['SCIPY-POWELL']['xtol']
            powell_ftol = optimizer_options['SCIPY-POWELL']['ftol']

            try:
                rslt = fmin_powell(opt_obj.crit_func, x_optim_free_scaled_start,
                    args, powell_xtol, powell_ftol, powell_maxiter,
                    powell_maxfun, disp=0)

                success = (rslt[5] not in [1, 2])
                message = 'Optimization terminated successfully.'
                if rslt[5] == 1:
                    message = 'Maximum number of function evaluations.'
                elif rslt[5] == 2:
                    message = 'Maximum number of iterations.'

            except MaxfunError:
                success = False
                message = 'Maximum number of iterations exceeded.'

        record_estimation_final(success, message)
        record_estimation_stop()

    elif request == 'simulate':

        # Draw draws for the simulation.
        periods_draws_sims = create_draws(num_periods, num_agents_sim,
            seed_sim, is_debug)

        # Draw standard normal deviates for the solution and evaluation step.
        periods_draws_emax = create_draws(num_periods, num_draws_emax,
            seed_emax, is_debug)

        # Collect arguments to pass in different implementations of the
        # simulation.
        periods_rewards_systematic, states_number_period, mapping_state_idx, \
            periods_emax, states_all = pyth_solve(coeffs_a, coeffs_b,
            coeffs_edu, coeffs_home, shocks_cholesky, is_interpolated,
            num_points_interp, num_draws_emax, num_periods, is_myopic,
            edu_start, is_debug, edu_max, min_idx, delta, periods_draws_emax,
            measure, level, file_sim, optimizer_options)

        solution = (periods_rewards_systematic, states_number_period,
            mapping_state_idx, periods_emax, states_all)

        data_array = pyth_simulate(periods_rewards_systematic,
            mapping_state_idx, periods_emax, states_all, shocks_cholesky,
            num_periods, edu_start, edu_max, delta, num_agents_sim,
            periods_draws_sims, seed_sim, file_sim)

        args = (solution, data_array)

    else:
        raise AssertionError

    return args


def get_precondition_matrix(preconditioning, paras_fixed,
        x_optim_all_unscaled_start, args, maxfun):
    """ Get the preconditioning matrix for the optimization.
    """
    # Auxiliary objects
    num_free = paras_fixed.count(False)

    # Set up a special instance of the optimization class.
    opt_obj = OptimizationClass()

    opt_obj.x_optim_all_unscaled_start = x_optim_all_unscaled_start
    opt_obj.precond_matrix = np.identity(num_free)
    opt_obj.paras_fixed = paras_fixed
    opt_obj.is_scaling = False

    # Distribute information about user request.
    precond_type, precond_minimum, precond_eps = preconditioning

    # Get the subset of free parameters for subsequent numerical
    # approximation of the gradient.
    x_optim_free_unscaled_start = []
    for i in range(27):
        if not paras_fixed[i]:
            x_optim_free_unscaled_start += [x_optim_all_unscaled_start[i]]

    precond_matrix = np.zeros((num_free, num_free))

    if precond_type == 'identity' or maxfun == 0:
        precond_matrix = np.identity(num_free)
    else:
        opt_obj.is_scaling = None
        grad = approx_fprime(x_optim_free_unscaled_start, opt_obj.crit_func,
            precond_eps, *args)

        for i in range(num_free):
            grad[i] = max(np.abs(grad[i]), precond_minimum)
            precond_matrix[i, i] = grad[i]

    return precond_matrix

