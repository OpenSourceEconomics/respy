from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import approx_fprime
from scipy.optimize import fmin_powell
from scipy.optimize import fmin_bfgs

from math import floor
from math import log10
import numpy as np

from respy.python.record.record_estimation import record_estimation_scalability
from respy.python.record.record_estimation import record_estimation_scaling
from respy.python.record.record_estimation import record_estimation_final
from respy.python.record.record_estimation import record_estimation_stop
from respy.python.solve.solve_auxiliary import pyth_create_state_space
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.estimate.estimate_wrapper import OptimizationClass
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.python.simulate.simulate_python import pyth_simulate
from respy.python.shared.shared_auxiliary import apply_scaling
from respy.python.shared.shared_auxiliary import create_draws
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.python.shared.shared_constants import NUM_PARAS
from respy.python.solve.solve_python import pyth_solve
from respy.custom_exceptions import MaxfunError


def respy_interface(respy_obj, request, data_array=None):
    """ This function provides the interface to the PYTHOn functionality.
    """
    # Distribute class attributes
    optim_paras, num_periods, edu_start, is_debug, edu_max, \
        num_draws_prob, seed_prob, num_draws_emax, seed_emax, \
        min_idx, is_myopic, is_interpolated, num_points_interp, maxfun, \
        optimizer_used, tau, optimizer_options, seed_sim, \
        num_agents_sim, ambi_spec, file_sim, precond_spec, type_spec, \
        num_types = dist_class_attributes(respy_obj, 'optim_paras',
            'num_periods', 'edu_start', 'is_debug', 'edu_max', 'num_draws_prob',
                'seed_prob', 'num_draws_emax', 'seed_emax', 'min_idx',
                'is_myopic', 'is_interpolated', 'num_points_interp', 'maxfun',
                'optimizer_used', 'tau', 'optimizer_options',
                'seed_sim', 'num_agents_sim', 'ambi_spec', 'file_sim',
                'precond_spec', 'type_spec', 'num_types')

    if request == 'estimate':

        periods_draws_prob = create_draws(num_periods, num_draws_prob,
            seed_prob, is_debug)

        # Draw standard normal deviates for the solution and evaluation step.
        periods_draws_emax = create_draws(num_periods, num_draws_emax,
            seed_emax, is_debug)

        # Construct starting values
        x_optim_free_unscaled_start = get_optim_paras(optim_paras, 'free',
            is_debug)

        x_optim_all_unscaled_start = get_optim_paras(optim_paras, 'all',
            is_debug)

        # Construct the state space
        states_all, states_number_period, mapping_state_idx, \
            max_states_period = pyth_create_state_space(num_periods,
                edu_start, edu_max, min_idx, num_types)

        # Cutting to size
        states_all = states_all[:, :max(states_number_period), :]

        # Collect arguments that are required for the criterion function. These
        # must be in the correct order already.
        args = (is_interpolated, num_draws_emax, num_periods,
            num_points_interp, is_myopic, edu_start, is_debug, edu_max,
            data_array, num_draws_prob, tau, periods_draws_emax,
            periods_draws_prob, states_all, states_number_period,
            mapping_state_idx, max_states_period, ambi_spec, type_spec,
            optimizer_options)

        # Special case where just an evaluation at the starting values is
        # requested is accounted for. Note, that the relevant value of the
        # criterion function is always the one indicated by the class
        # attribute and not the value returned by the optimization algorithm.
        num_free = optim_paras['paras_fixed'].count(False)

        paras_bounds_free_unscaled = []
        for i in range(NUM_PARAS):
            if not optim_paras['paras_fixed'][i]:
                lower, upper = optim_paras['paras_bounds'][i][:]
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

        precond_matrix = get_precondition_matrix(precond_spec,
            optim_paras, x_optim_all_unscaled_start, args, maxfun)

        x_optim_free_scaled_start = apply_scaling(x_optim_free_unscaled_start,
            precond_matrix, 'do')

        paras_bounds_free_scaled = np.tile(np.nan, (num_free, 2))
        for i in range(2):
            paras_bounds_free_scaled[:, i] = apply_scaling(
                paras_bounds_free_unscaled[:, i], precond_matrix, 'do')

        record_estimation_scaling(x_optim_free_unscaled_start,
            x_optim_free_scaled_start, paras_bounds_free_scaled,
            precond_matrix, optim_paras['paras_fixed'])

        opt_obj = OptimizationClass()
        opt_obj.maxfun = maxfun
        opt_obj.paras_fixed = optim_paras['paras_fixed']
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
            periods_emax, states_all = pyth_solve(is_interpolated,
            num_points_interp, num_draws_emax, num_periods, is_myopic,
            edu_start, is_debug, edu_max, min_idx, periods_draws_emax,
            ambi_spec, optim_paras, file_sim, optimizer_options, type_spec,
            num_types)

        solution = (periods_rewards_systematic, states_number_period,
            mapping_state_idx, periods_emax, states_all)

        data_array = pyth_simulate(periods_rewards_systematic,
            mapping_state_idx, periods_emax, states_all, num_periods, edu_start,
            edu_max, num_agents_sim, periods_draws_sims, seed_sim, file_sim,
            optim_paras, num_types, type_spec, is_debug)

        args = (solution, data_array)

    else:
        raise AssertionError

    return args


def get_precondition_matrix(precond_spec, optim_paras,
        x_optim_all_unscaled_start, args, maxfun):
    """ Get the preconditioning matrix for the optimization.
    """
    # Auxiliary objects
    num_free = optim_paras['paras_fixed'].count(False)

    # Set up a special instance of the optimization class.
    opt_obj = OptimizationClass()

    opt_obj.x_optim_all_unscaled_start = x_optim_all_unscaled_start
    opt_obj.precond_matrix = np.identity(num_free)
    opt_obj.paras_fixed = optim_paras['paras_fixed']
    opt_obj.is_scaling = False

    # Distribute information about user request.
    precond_minimum = precond_spec['minimum']
    precond_type = precond_spec['type']
    precond_eps = precond_spec['eps']

    # Get the subset of free parameters for subsequent numerical
    # approximation of the gradient.
    x_optim_free_unscaled_start = []
    for i in range(NUM_PARAS):
        if not optim_paras['paras_fixed'][i]:
            x_optim_free_unscaled_start += [x_optim_all_unscaled_start[i]]

    if precond_type == 'identity' or maxfun == 0:
        precond_matrix = np.identity(num_free)
    elif precond_type == 'magnitudes':
        precond_matrix = get_scales_magnitudes(x_optim_free_unscaled_start)
    elif precond_type == 'gradient':
        opt_obj.is_scaling = None
        grad = approx_fprime(x_optim_free_unscaled_start, opt_obj.crit_func,
            precond_eps, *args)

        precond_matrix = np.zeros((num_free, num_free))
        for i in range(num_free):
            grad[i] = max(np.abs(grad[i]), precond_minimum)
            precond_matrix[i, i] = grad[i]
    else:
        raise NotImplementedError

    # Write out scaling matrix to allow for restart.
    np.savetxt('scaling.respy.out', precond_matrix, fmt='%45.15f')

    return precond_matrix


def get_scales_magnitudes(x_optim_free_unscaled_start):
    """ This function calculated the scaling factor based on the different
    magnitudes of the variables.
    """
    # Auxiliary objects
    num_free = len(x_optim_free_unscaled_start)

    # Initialize container
    precond_matrix = np.zeros((num_free, num_free))

    for i, x in enumerate(x_optim_free_unscaled_start):
        # Special case
        if x == 0.0:
            scale = 1
        else:
            magnitude = int(floor(log10(abs(x))))
            if magnitude == 0:
                scale = 1.0 / 10.0
            else:
                scale = (10 ** magnitude) ** (-1) / 10.0
        precond_matrix[i, i] = scale

    return precond_matrix

