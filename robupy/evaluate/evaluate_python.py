""" This module provides the interface to the functionality needed to
evaluate the likelihood function.
"""

# project library
from robupy.evaluate.evaluate_auxiliary import evaluate_criterion_function
from robupy.solve.solve_python import solve_python

''' Main function
'''


def evaluate_python(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        shocks_cholesky, is_deterministic, is_interpolated, num_draws_emax,
        periods_draws_emax, is_ambiguous, num_periods, num_points, edu_start,
        is_myopic, is_debug, measure, edu_max, min_idx, delta, level,
        data_frame,  num_agents, num_draws_prob,
        periods_draws_prob, is_python):
    """ Evaluate the criterion function of the model using PYTHON/F2PY code.
    This purpose of this wrapper is to extract all relevant information from
    the project class to pass it on to the actual evaluation functions. This is
    required to align the functions across the PYTHON and F2PY implementations.
    """


    # Transform dataset to array for easy access
    data_array = data_frame.as_matrix()

    # Solve model for given parametrization
    args = solve_python(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        shocks_cholesky, is_deterministic, is_interpolated, num_draws_emax,
        periods_draws_emax, is_ambiguous, num_periods, num_points, edu_start,
        is_myopic, is_debug, measure, edu_max, min_idx, delta, level,
        is_python)

    # Distribute return arguments from solution run
    periods_payoffs_systematic, periods_payoffs_ex_post = args[:2]
    mapping_state_idx, periods_emax, states_all = args[4:7]

    # Evaluate the criterion function
    if is_python:
        likl = evaluate_criterion_function(mapping_state_idx, periods_emax,
            periods_payoffs_systematic, states_all, shocks_cov, edu_max, delta,
            edu_start, num_periods, shocks_cholesky, num_agents, num_draws_prob,
            data_array, periods_draws_prob, is_deterministic)
    else:
        import robupy.fortran.f2py_library as f2py_library
        likl = f2py_library.wrapper_evaluate_criterion_function(
            mapping_state_idx, periods_emax, periods_payoffs_systematic,
            states_all, shocks_cov, edu_max, delta, edu_start, num_periods,
            shocks_cholesky, num_agents, num_draws_prob, data_array,
            periods_draws_prob, is_deterministic)

    # Finishing
    return likl


