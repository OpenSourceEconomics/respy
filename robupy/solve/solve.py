""" This module contains the interface to solve the model.
"""

# standard library

# project library
from robupy.fortran.f2py_library import f2py_create_state_space
from robupy.fortran.f2py_library import f2py_solve

from robupy.fortran.fortran import fort_solve
from robupy.shared.auxiliary import add_solution
from robupy.shared.auxiliary import create_draws
from robupy.shared.auxiliary import distribute_class_attributes
from robupy.shared.auxiliary import distribute_model_paras
from robupy.shared.auxiliary import replace_missing_values
from robupy.solve.solve_auxiliary import stop_logging, start_logging, \
    summarize_ambiguity, cleanup, _start_ambiguity_logging, check_input
from robupy.solve.solve_python import pyth_solve

''' Main function
'''


def solve(robupy_obj):
    """ Solve dynamic programming problem by backward induction.
    """
    # Checks, cleanup, start logger
    assert check_input(robupy_obj)

    cleanup()

    start_logging()

    # Distribute class attributes
    model_paras, num_periods, edu_start, is_debug, edu_max, delta, \
        is_deterministic, version, num_draws_emax, seed_emax, is_interpolated, \
        is_ambiguous, num_points, is_myopic, min_idx, measure, level, store = \
            distribute_class_attributes(robupy_obj,
                'model_paras', 'num_periods', 'edu_start', 'is_debug',
                'edu_max', 'delta', 'is_deterministic', 'version',
                'num_draws_emax', 'seed_emax', 'is_interpolated',
                'is_ambiguous', 'num_points', 'is_myopic', 'min_idx', 'measure',
                'level', 'store')

    # Construct auxiliary objects
    _start_ambiguity_logging(is_ambiguous, is_debug)

    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
        distribute_model_paras(model_paras, is_debug)

    # Get the relevant set of disturbances. These are standard normal draws
    # in the case of an ambiguous world. This function is located outside
    # the actual bare solution algorithm to ease testing across
    # implementations.
    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax,
        is_debug, 'emax', shocks_cholesky)

    # Collect arguments
    pyth_args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
        num_periods, num_points, is_myopic, edu_start, is_debug, measure,
        edu_max, min_idx, delta, level, shocks_cholesky, periods_draws_emax)

    fort_args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
        is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
        num_periods, num_points, is_myopic, edu_start, seed_emax,
        is_debug, measure, edu_max, min_idx, delta, level)

    # Select appropriate interface. The additional preparations for the F2PY
    # interface are required as only explicit shape arguments can be passed
    # into the interface.
    if version == 'FORTRAN':
        solution = fort_solve(*fort_args)
    elif version == 'PYTHON':
        solution = pyth_solve(*pyth_args)
    elif version == 'F2PY':
        args = (num_periods, edu_start, edu_max, min_idx)
        max_states_period = f2py_create_state_space(*args)[3]
        solution = f2py_solve(*pyth_args + (max_states_period,))
    else:
        raise NotImplementedError

    # Replace missing values
    solution = replace_missing_values(solution)
#
    # Attach solution to class instance
    robupy_obj = add_solution(robupy_obj, store, *solution)

    # Summarize optimizations in case of ambiguity
    if is_debug and is_ambiguous and (not is_myopic):
        summarize_ambiguity(robupy_obj)

    # Orderly shutdown of logging capability.
    stop_logging()

    # Finishing
    return robupy_obj


