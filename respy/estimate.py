# project library
from respy.python.estimate.estimate_auxiliary import get_optim_paras
from respy.python.estimate.estimate_auxiliary import check_input

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.shared.shared_auxiliary import create_draws

from respy.python.estimate.estimate_wrapper import OptimizationClass

from respy.process import process
from respy.fortran.fortran import resfort_interface


def estimate(respy_obj):
    """ Estimate the model
    """
    # Read in estimation dataset. It only reads in the number of agents
    # requested for the estimation.
    data_frame = process(respy_obj)

    # Antibugging
    assert check_input(respy_obj, data_frame)

    # Distribute class attributes
    model_paras, num_periods, num_agents_est, edu_start, is_debug, edu_max, \
        delta, num_draws_prob, seed_prob, num_draws_emax, seed_emax, min_idx,\
        is_myopic, is_interpolated, num_points_interp, version, maxiter, \
        optimizer_used, tau, paras_fixed, optimizer_options, is_parallel, \
        num_procs = \
            dist_class_attributes(respy_obj,
                'model_paras', 'num_periods', 'num_agents_est', 'edu_start',
                'is_debug', 'edu_max', 'delta', 'num_draws_prob', 'seed_prob',
                'num_draws_emax', 'seed_emax', 'min_idx', 'is_myopic',
                'is_interpolated', 'num_points_interp', 'version', 'maxiter',
                'optimizer_used', 'tau', 'paras_fixed', 'optimizer_options',
                'is_parallel', 'num_procs')

    # Auxiliary objects
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = \
        dist_model_paras(model_paras, is_debug)

    # Draw standard normal deviates for the solution and evaluation step.
    periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob,
        is_debug)

    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax,
        is_debug)

    # Construct starting values
    x_free_start = get_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cholesky, 'free', paras_fixed, is_debug)

    x_all_start = get_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cholesky, 'all', paras_fixed, is_debug)

    data_array = data_frame.as_matrix()

    # Collect arguments that are required for the criterion function. These
    # must be in the correct order already.
    args = (is_interpolated, num_draws_emax, num_periods, num_points_interp,
        is_myopic, edu_start, is_debug, edu_max, min_idx, delta, data_array,
        num_agents_est, num_draws_prob, tau, periods_draws_emax,
        periods_draws_prob)

    # Setup optimization class, which handles all the details depending on the
    # request.
    if version in ['PYTHON']:
        opt_obj = OptimizationClass()

        opt_obj.set_attr('args', args)

        opt_obj.set_attr('optimizer_options', optimizer_options)

        opt_obj.set_attr('x_info', (x_all_start, paras_fixed))

        opt_obj.set_attr('optimizer_used', optimizer_used)

        opt_obj.set_attr('version', version)

        opt_obj.set_attr('maxiter', maxiter)

        opt_obj.lock()

        # Perform optimization.
        x, val = opt_obj.optimize(x_free_start)

    elif version in ['FORTRAN']:

        val = resfort_interface(respy_obj, 'estimate', data_array)

        x = x_all_start

    else:

        raise NotImplementedError
    # Finishing
    return x, val




