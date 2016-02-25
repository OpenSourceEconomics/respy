""" This modules contains some additional tests that are only used in long-run
development tests.
"""

# TODO: I am in a development push the version constraints need to be
# TODO: revisited later.

# standard library
from pandas.util.testing import assert_frame_equal
import statsmodels.api as sm

from scipy.optimize.slsqp import _minimize_slsqp
from scipy.optimize import approx_fprime
from scipy.optimize import rosen_der
from scipy.optimize import rosen

import pandas as pd
import numpy as np

import scipy
import sys
import os

# testing library
from modules.auxiliary import transform_robupy_to_restud
from modules.auxiliary import write_interpolation_grid
from modules.auxiliary import write_disturbances
from modules.auxiliary import build_f2py_testing
from modules.auxiliary import compile_package
from modules.auxiliary import cleanup

# ROBUPY import
sys.path.insert(0, os.environ['ROBUPY'])
from robupy import simulate
from robupy import read
from robupy import solve

from robupy.python.py.python_library import _get_simulated_indicator
from robupy.python.py.python_library import _get_exogenous_variables
from robupy.python.py.python_library import _get_endogenous_variable
from robupy.python.py.python_library import _get_predictions
from robupy.python.py.python_library import get_payoffs

from robupy.tests.random_init import generate_random_dict
from robupy.tests.random_init import print_random_dict
from robupy.tests.random_init import generate_init

from robupy.python.py.ambiguity import get_payoffs_ambiguity
from robupy.python.py.auxiliary import simulate_emax
from robupy.python.py.ambiguity import _divergence
from robupy.python.py.ambiguity import _criterion


from robupy.auxiliary import replace_missing_values
from robupy.auxiliary import create_disturbances

''' Main
'''


def test_85():
    """ This test compares the simulated datasets for solutions that involve
    interpolation in PYTHON and FORTRAN versions.
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)

    # Constraints
    constraints = dict()
    constraints['debug'] = True
    constraints['measure'] = 'kl'

    # This ensures that interpolation is actually run.
    num_periods = np.random.random_integers(5, 10)
    num_points = np.random.random_integers(40, 54)

    constraints['periods'] = num_periods
    constraints['points'] = num_points
    constraints['apply'] = True

    # We need to ensure that the number of agents is less or equal than the
    # number of draws so that we can standardize the disturbances across
    # implementations.
    num_draws = np.random.random_integers(1, 100)
    num_agents = np.random.random_integers(1, num_draws)

    constraints['agents'] = num_agents
    constraints['draws'] = num_draws

    # Just making sure that it also works for this special case. Note that
    # this special case is currently only working in the risk case.
    if np.random.choice([True, False, False, False]):
        constraints['level'] = 0.00
        constraints['eps_zero'] = True

    # Generate random initialization file.
    init_dict = generate_random_dict(constraints)

    # Write out disturbances to align the three implementations.
    write_disturbances(init_dict)

    # Test the standardization across PYTHON, F2PY, and FORTRAN
    # implementations. This is possible as we write out an interpolation
    # grid to disk which is used for both functions. This only works if
    # IS_DEBUG is set to true.
    print_random_dict(init_dict)

    robupy_obj = read('test.robupy.ini')

    solve(robupy_obj)

    states_number_period = robupy_obj.get_attr('states_number_period')

    write_interpolation_grid(num_periods, num_points, states_number_period)

    # Initialize containers
    base = None
    for version in ['PYTHON', 'FORTRAN', 'F2PY']:

        # This ensures that the optimized version agrees with all other
        # implementations as well.
        if version in ['OPTIMIZATION']:
            compile_package('--fortran --debug --optimization', True)
            version = 'FORTRAN'

        # Prepare initialization file
        init_dict['PROGRAM']['version'] = version

        print_random_dict(init_dict)

        # Simulate the ROBUPY package
        robupy_obj = read('test.robupy.ini')

        solve(robupy_obj)

        # Load simulated data frame
        data_frame = pd.read_csv('data.robupy.dat', delim_whitespace=True)

        data_frame.to_csv(version + '.dat')

        # Compare
        if base is None:
            base = data_frame.copy()

        assert_frame_equal(base, data_frame)

    # Cleanup
    cleanup()


def test_86():
    """ Further tests for the interpolation routines.
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)

    # Load interface to debugging library
    import robupy.python.f2py.f2py_debug as fort

    for k in range(10):

        # Impose constraints. The FORTRAN code does not include other
        # measures just yet.
        constraints = dict()
        constraints['measure'] = 'kl'

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        # Extract ingredients for interface
        periods_payoffs_systematic = robupy_obj.get_attr('periods_payoffs_systematic')

        states_number_period = robupy_obj.get_attr('states_number_period')

        mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')

        periods_emax = robupy_obj.get_attr('periods_emax')

        num_periods = robupy_obj.get_attr('num_periods')

        states_all = robupy_obj.get_attr('states_all')

        num_points = robupy_obj.get_attr('num_points')

        edu_start = robupy_obj.get_attr('edu_start')

        num_draws = robupy_obj.get_attr('num_draws')

        is_debug = robupy_obj.get_attr('is_debug')

        measure = robupy_obj.get_attr('measure')

        edu_max = robupy_obj.get_attr('edu_max')

        shocks = robupy_obj.get_attr('shocks')

        delta = robupy_obj.get_attr('delta')

        level = robupy_obj.get_attr('level')

        # Add some additional objects required for the interfaces to the
        # functions.
        period = np.random.choice(range(num_periods))

        periods_eps_relevant = create_disturbances(robupy_obj, False)

        eps_relevant = periods_eps_relevant[period, :, :]

        num_states = states_number_period[period]

        shifts = np.random.randn(4)

        # Slight modification of request which assures that the interpolation
        # code is working.
        num_points = min(num_points, num_states)

        # Get the IS_SIMULATED indicator for the subset of points which are
        # used for the predication model. The integrity of the corresponding
        # FORTRAN function is tested in test_88().
        args = (num_points, num_states, period, num_periods, is_debug)
        is_simulated = _get_simulated_indicator(*args)

        # Construct the exogenous variables for all points of the state space.
        args = [period, num_periods, num_states, delta,
                periods_payoffs_systematic, shifts, edu_max, edu_start,
                mapping_state_idx, periods_emax, states_all]

        py = _get_exogenous_variables(*args)
        f90 = fort.wrapper_get_exogenous_variables(*args)

        np.testing.assert_equal(py, f90)

        # Distribute validated results for further functions.
        exogenous, maxe = py

        # Construct endogenous variable so that the prediction model can be
        # fitted.
        args += [is_simulated, num_draws, shocks, level, is_debug, measure,
                 maxe, eps_relevant]
        py = _get_endogenous_variable(*args)
        f90 = fort.wrapper_get_endogenous_variable(*args)

        np.testing.assert_equal(py, replace_missing_values(f90))

        # Distribute validated results for further functions.
        endogenous = py

        # Get predictions for expected future values
        args = [endogenous, exogenous, maxe, is_simulated, num_points,
                num_states, is_debug]

        py, _ = _get_predictions(*args)
        f90 = fort.wrapper_get_predictions(*args[:-1])

        np.testing.assert_array_almost_equal(py, f90)

    # Cleanup
    cleanup()


def test_87():
    """ This function compares the results from the payoff functions across
    implementations.
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)

    # Load interface to debugging library
    import robupy.python.f2py.f2py_debug as fort

    for _ in range(10):

        # Impose constraints
        constraints = dict()
        constraints['measure'] = 'kl'

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        # Extract ingredients for interface
        periods_payoffs_systematic = robupy_obj.get_attr('periods_payoffs_systematic')

        states_number_period = robupy_obj.get_attr('states_number_period')

        mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')

        periods_emax = robupy_obj.get_attr('periods_emax')

        num_periods = robupy_obj.get_attr('num_periods')

        states_all = robupy_obj.get_attr('states_all')

        num_draws = robupy_obj.get_attr('num_draws')

        edu_start = robupy_obj.get_attr('edu_start')

        is_debug = robupy_obj.get_attr('is_debug')

        edu_max = robupy_obj.get_attr('edu_max')

        shocks = robupy_obj.get_attr('shocks')

        measure = robupy_obj.get_attr('measure')

        delta = robupy_obj.get_attr('delta')

        level = robupy_obj.get_attr('level')

        # Iterate over a couple of admissible points
        for _ in range(10):

            # Select random points
            period = np.random.choice(range(num_periods))
            k = np.random.choice(range(states_number_period[period]))

            # Finalize extraction of ingredients
            payoffs_systematic = periods_payoffs_systematic[period, k, :]
            eps_relevant = np.random.sample((num_draws, 4))

            # Extract payoffs using PYTHON and FORTRAN codes.
            py = get_payoffs(num_draws, eps_relevant, period, k,
                    payoffs_systematic, edu_max, edu_start, mapping_state_idx,
                    states_all, num_periods, periods_emax, delta, is_debug,
                    shocks, level, measure)

            f90 = fort.wrapper_get_payoffs(num_draws, eps_relevant, period, k,
                    payoffs_systematic, edu_max, edu_start, mapping_state_idx,
                    states_all, num_periods, periods_emax, delta, is_debug,
                    shocks, level, measure)

            # Compare returned array on expected future values, ex post
            # payoffs, and future payoffs.
            for i in range(3):
                np.testing.assert_array_almost_equal(py[i], f90[i])

    # Cleanup
    cleanup()


def test_88():
    """ This is a special test for auxiliary functions related to the
    interpolation setup.
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)

    # Load interface to debugging library
    import robupy.python.f2py.f2py_debug as fort

    for _ in range(1000):

        # Cleanup
        try:
            os.unlink('interpolation.txt')
        except FileNotFoundError:
            pass

        # Draw random request for testing
        num_states = np.random.random_integers(1, 500)
        num_points = np.random.random_integers(1, num_states)
        candidates = list(range(num_states))

        num_periods = np.random.random_integers(4, 10)
        period = np.random.random_integers(1, num_periods - 1)

        states_number_period = np.random.random_integers(1, num_states,
                                                         size=num_periods)

        # The number of states has to part of this list for the tests to run.
        idx = np.random.choice(range(num_periods))
        states_number_period[idx] = num_states

        is_debug = np.random.choice([True, False])

        # Check function for random choice and make sure that there are no
        # duplicates.
        f90 = fort.wrapper_random_choice(candidates, num_states, num_points)
        np.testing.assert_equal(len(set(f90)), len(f90))
        np.testing.assert_equal(len(f90), num_points)

        # Check the standard cases of the function.
        args = (num_points, num_states, period, num_periods, is_debug)
        f90 = fort.wrapper_get_simulated_indicator(*args)

        np.testing.assert_equal(len(f90), num_states)
        np.testing.assert_equal(np.all(f90) in [0, 1], True)

        # Test the standardization across PYTHON, F2PY, and FORTRAN
        # implementations. This is possible as we write out an interpolation
        # grid to disk which is used for both functions. This only works if
        # IS_DEBUG is set to true.
        write_interpolation_grid(num_periods, num_points, states_number_period)

        args = (num_points, num_states, period, num_periods, True)
        py = _get_simulated_indicator(*args)
        f90 = fort.wrapper_get_simulated_indicator(*args)

        np.testing.assert_array_equal(f90, 1*py)
        os.unlink('interpolation.txt')

        # Special case where number of interpolation points are same as the
        # number of candidates. In that case the returned indicator should be
        # all TRUE.
        args = (num_states, num_states, period, num_periods, True)
        f90 = fort.wrapper_get_simulated_indicator(*args)
        np.testing.assert_equal(sum(f90), num_states)

    # Cleanup
    cleanup()


def test_89():
    """ This is the special case where the EMAX better be equal to the MAXE.
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)

    # Set initial constraints
    constraints = dict()
    constraints['apply'] = False
    constraints['level'] = 0.00
    constraints['periods'] = np.random.random_integers(2, 6)
    constraints['eps_zero'] = True

    # Initialize request
    init_dict = generate_random_dict(constraints)
    baseline = None

    # Solve with and without interpolation code
    for _ in range(2):

        # Write out request
        print_random_dict(init_dict)

        # Process and solve
        robupy_obj = read('test.robupy.ini')
        robupy_obj = solve(robupy_obj)

        # Extract results
        states_number_period = robupy_obj.get_attr('states_number_period')
        emax = robupy_obj.get_attr('periods_emax')

        # Store and check results
        if baseline is None:
            baseline = emax
        else:
            np.testing.assert_array_almost_equal(baseline, emax)

        # Updates for second iteration. This ensures that there is at least
        # one interpolation taking place.
        init_dict['INTERPOLATION']['points'] = max(states_number_period) - 1
        init_dict['INTERPOLATION']['apply'] = True

    # Cleanup
    cleanup()


def test_90():
    """ This test compares the results from a solution using the
    interpolation code for the special case where the number of interpolation
    points is exactly the number of states in the final period. In this case
    the interpolation code is run and then all predicted values replaced
    with their actual values.
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)

    # Set initial constraints
    constraints = dict()
    constraints['apply'] = False

    # Initialize request
    init_dict = generate_random_dict(constraints)
    baseline = None

    # Solve with and without interpolation code
    for _ in range(2):

        # Write out request
        print_random_dict(init_dict)

        # Process and solve
        robupy_obj = read('test.robupy.ini')
        robupy_obj = solve(robupy_obj)

        # Extract results
        states_number_period = robupy_obj.get_attr('states_number_period')
        emax = robupy_obj.get_attr('periods_emax')

        # Store and check results
        if baseline is None:
            baseline = emax
        else:
            np.testing.assert_array_almost_equal(baseline, emax)

        # Updates for second iteration
        init_dict['INTERPOLATION']['points'] = max(states_number_period)
        init_dict['INTERPOLATION']['apply'] = True

    # Cleanup
    cleanup()


def test_91():
    """ This test compares the expected future payoffs to the ex post payoffs
    in the special case, where agents are myopic and there is no randomness
    in the disturbances (all realizations are set to zero).
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)

    # Iterate over random test cases
    for _ in range(10):

        # Generate constraint periods
        constraints = dict()
        constraints['level'] = 0.00
        constraints['delta'] = 0.00
        constraints['eps_zero'] = True
        constraints['periods'] = np.random.random_integers(2, 5)

        # Ex post payoffs are not available for periods where interpolation
        # is set up.
        constraints['apply'] = False

        # Sample a random estimation request and write it to disk.
        init_dict = generate_random_dict(constraints)
        print_random_dict(init_dict)

        # Run interpolation routine
        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        # Extract relevant information
        payoffs_ex_post = robupy_obj.get_attr('periods_payoffs_ex_post')

        emax = robupy_obj.get_attr('periods_emax')

        # Collapse to maximum values in each period
        emax_myopic = np.amax(payoffs_ex_post, axis=2)

        # Ensure equivalence
        np.testing.assert_array_almost_equal(emax_myopic, emax)

    # Cleanup
    cleanup()


def test_92():
    """ This test compares the functions calculating the payoffs under
    ambiguity.
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)
    import robupy.python.f2py.f2py_debug as fort

    # Iterate over random test cases
    for _ in range(10):

        # Generate constraint periods
        constraints = dict()
        constraints['level'] = 0.0
        constraints['version'] = 'PYTHON'
        constraints['measure'] = 'kl'

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        # Extract relevant information
        periods_payoffs_systematic = robupy_obj.get_attr('periods_payoffs_systematic')

        states_number_period = robupy_obj.get_attr('states_number_period')

        mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')

        periods_emax = robupy_obj.get_attr('periods_emax')

        num_periods = robupy_obj.get_attr('num_periods')

        states_all = robupy_obj.get_attr('states_all')

        num_draws = robupy_obj.get_attr('num_draws')

        edu_start = robupy_obj.get_attr('edu_start')

        edu_max = robupy_obj.get_attr('edu_max')

        measure = robupy_obj.get_attr('measure')

        shocks = robupy_obj.get_attr('shocks')

        delta = robupy_obj.get_attr('delta')

        debug = False

        # Sample disturbances
        eps_standard = np.random.multivariate_normal(np.zeros(4),
                        np.identity(4), (num_draws,))

        # Sampling of random period and admissible state index
        period = np.random.choice(range(num_periods))
        k = np.random.choice(range(states_number_period[period]))

        # Select systematic payoffs
        payoffs_systematic = periods_payoffs_systematic[period, k, :]

        # Set up optimization task
        level = np.random.uniform(0.01, 1.00)

        args = [num_draws, eps_standard, period, k, payoffs_systematic,
            edu_max, edu_start, mapping_state_idx, states_all, num_periods,
            periods_emax, debug, delta, shocks, level, measure]

        f = fort.wrapper_get_payoffs_ambiguity(*args)[0]
        py = get_payoffs_ambiguity(*args)[0]

        np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)

    # Cleanup
    cleanup()


def test_93():
    """ This test case compares the results from the SLSQP implementations in
    PYTHON and FORTRAN for the actual optimization problem.
    """
    # Ensure interface is available
    build_f2py_testing(is_hidden=True)
    import modules.f2py_testing as fort

    # Sample problem parameters
    for _ in range(10):

        maxiter = np.random.random_integers(1, 100)
        ftol = np.random.uniform(0.000000, 1e-5)
        x0 = np.random.normal(size=2)

        eps = 1e-6

        shocks = np.identity(4)*np.random.normal(size=1)**2
        level = np.random.normal(size=1)**2

        # Setting up PYTHON SLSQP interface for constraints
        constraint = dict()
        constraint['type'] = 'eq'
        constraint['args'] = (shocks, level)
        constraint['fun'] = _divergence

        # Generate constraint periods
        constraints = dict()
        constraints['version'] = 'PYTHON'

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        # Extract relevant information
        periods_payoffs_systematic = robupy_obj.get_attr('periods_payoffs_systematic')

        states_number_period = robupy_obj.get_attr('states_number_period')

        mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')

        periods_emax = robupy_obj.get_attr('periods_emax')

        num_periods = robupy_obj.get_attr('num_periods')

        states_all = robupy_obj.get_attr('states_all')

        num_draws = robupy_obj.get_attr('num_draws')

        edu_start = robupy_obj.get_attr('edu_start')

        edu_max = robupy_obj.get_attr('edu_max')

        delta = robupy_obj.get_attr('delta')

        is_debug = False

        # Sample disturbances
        eps_standard = np.random.multivariate_normal(np.zeros(4),
                                np.identity(4), (num_draws,))

        # Sampling of random period and admissible state index
        period = np.random.choice(range(num_periods))
        k = np.random.choice(range(states_number_period[period]))

        # Select systematic payoffs
        payoffs_systematic = periods_payoffs_systematic[period, k, :]

        args = (num_draws, eps_standard, period, k, payoffs_systematic, edu_max,
            edu_start, mapping_state_idx, states_all, num_periods, periods_emax,
            delta)

        opt = _minimize_slsqp(_criterion, x0, args, maxiter=maxiter,
                       ftol=ftol, constraints=constraint)

        # Stabilization. This is done as part of the fortran implementation.
        if opt['success']:
            py = opt['x']
        else:
            py = x0

        f = fort.wrapper_slsqp_robufort(x0, maxiter, ftol, eps, num_draws,
                eps_standard, period, k, payoffs_systematic, edu_max, edu_start,
                mapping_state_idx, states_all, num_periods, periods_emax,
                delta, is_debug, shocks, level)

        # Check equality. If not equal up to the tolerance, also check
        # whether the result from the FORTRAN implementation is even better.
        try:
            np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)
        except AssertionError:
            if _criterion(f, *args) < _criterion(py, *args):
                pass
            else:
                raise AssertionError

    # Cleanup
    cleanup()


def test_94():
    """ This test case compare the results of a debugging setup for the SLSQP
    algorithm's PYTHON and FORTRAN implementation
    """
    # Ensure interface is available
    build_f2py_testing(is_hidden=True)
    import modules.f2py_testing as fort

    # Sample basic test case
    maxiter = np.random.random_integers(1, 100)
    num_dim = np.random.random_integers(2, 4)
    ftol = np.random.uniform(0.000000, 1e-5)
    x0 = np.random.normal(size=num_dim)

    # Evaluation of Rosenbrock function. We are using the FORTRAN version
    # in the development of the optimization routines.
    f90 = fort.wrapper_debug_criterion_function(x0, num_dim)
    py = rosen(x0)
    np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

    py = rosen_der(x0)
    f90 = fort.wrapper_debug_criterion_derivative(x0, len(x0))
    np.testing.assert_allclose(py, f90[:-1], rtol=1e-05, atol=1e-06)

    # Test the FORTRAN codes against the PYTHON implementation. This is
    # expected to fail sometimes due to differences in precision between the
    # two implementations. In particular, as updating steps of the optimizer
    # are very sensitive to just small differences in the derivative
    # information. The same functions are available as a FORTRAN
    # implementations.
    def debug_constraint_derivative(x):
        return np.ones(len(x))
    def debug_constraint_function(x):
        return np.sum(x) - 10.0

    # Setting up PYTHON SLSQP interface for constraints
    constraint = dict()
    constraint['type'] = 'eq'
    constraint['args'] = ()
    constraint['fun'] = debug_constraint_function
    constraint['jac'] = debug_constraint_derivative

    # Evaluate both implementations
    f = fort.wrapper_slsqp_debug(x0, maxiter, ftol, num_dim)
    py = _minimize_slsqp(rosen, x0, jac=rosen_der, maxiter=maxiter,
            ftol=ftol,  constraints=constraint)['x']
    np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)

    # Cleanup
    cleanup()


def test_95():
    """ Compare the evaluation of the criterion function for the ambiguity
    optimization and the simulated expected future value between the FORTRAN
    and PYTHON implementations. These tests are set up a separate test case
    due to the large setup cost to construct the ingredients for the interface.
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)

    import robupy.python.f2py.f2py_debug as fort

    for _ in range(10):

        # Generate constraint periods
        constraints = dict()
        constraints['version'] = 'PYTHON'

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        # Extract relevant information
        periods_payoffs_systematic = robupy_obj.get_attr('periods_payoffs_systematic')

        states_number_period = robupy_obj.get_attr('states_number_period')

        mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')

        periods_emax = robupy_obj.get_attr('periods_emax')

        num_periods = robupy_obj.get_attr('num_periods')

        states_all = robupy_obj.get_attr('states_all')

        num_draws = robupy_obj.get_attr('num_draws')

        edu_start = robupy_obj.get_attr('edu_start')

        edu_max = robupy_obj.get_attr('edu_max')

        delta = robupy_obj.get_attr('delta')

        # Sample disturbances
        eps_standard = np.random.multivariate_normal(np.zeros(4),
                            np.identity(4), (num_draws,))

        # Sampling of random period and admissible state index
        period = np.random.choice(range(num_periods))
        k = np.random.choice(range(states_number_period[period]))

        # Select systematic payoffs
        payoffs_systematic = periods_payoffs_systematic[period, k, :]

        # Evaluation point
        x = np.random.random(size=2)

        # Evaluation of simulated expected future values
        py, _, _ = simulate_emax(num_periods, num_draws, period, k,
            eps_standard, payoffs_systematic, edu_max, edu_start,
            periods_emax, states_all, mapping_state_idx, delta)

        f90, _, _ = fort.wrapper_simulate_emax(num_periods, num_draws,
            period, k, eps_standard, payoffs_systematic, edu_max,
            edu_start, periods_emax, states_all, mapping_state_idx, delta)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Criterion function for the determination of the worst case outcomes
        args = (num_draws, eps_standard, period, k, payoffs_systematic,
                edu_max, edu_start, mapping_state_idx, states_all, num_periods,
                periods_emax, delta)

        py = _criterion(x, *args)
        f90 = fort.wrapper_criterion(x, *args)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Evaluation of derivative of criterion function
        eps = np.random.uniform(0.000000, 0.5)

        py = approx_fprime(x, _criterion, eps, *args)
        f90 = fort.wrapper_criterion_approx_gradient(x, eps, *args)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

    # Cleanup
    cleanup()


def test_96():
    """ Compare results between FORTRAN and PYTHON of selected
    hand-crafted functions. In test_97() we test FORTRAN implementations
    against PYTHON intrinsic routines.
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)

    import robupy.python.f2py.f2py_debug as fort_debug
    import robupy.python.f2py.f2py_library as fort_lib
    import robupy.python.py.python_library as py_lib

    for _ in range(1000):

        # Draw random request for testing purposes
        matrix = (np.random.multivariate_normal(np.zeros(4), np.identity(
            4), 4))
        cov = np.dot(matrix, matrix.T)
        x = np.random.rand(2)
        level = np.random.random(1)
        eps = np.random.rand()**2

        # Kullback-Leibler (KL) divergence
        py = _divergence(x, cov, level)
        f90 = fort_debug.wrapper_divergence(x, cov, level)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Gradient approximation of KL divergence
        py = approx_fprime(x, _divergence, eps, cov, level)
        f90 = fort_debug.wrapper_divergence_approx_gradient(x, cov, level, eps)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

    for _ in range(25):

        # Create grid of admissible state space values.
        num_periods = np.random.random_integers(1, 15)
        edu_start = np.random.random_integers(1, 5)
        edu_max = edu_start + np.random.random_integers(1, 5)

        # Prepare interface
        min_idx = min(num_periods, (edu_max - edu_start + 1))

        # FORTRAN
        fort_a, fort_b, fort_c = fort_lib.wrapper_create_state_space(
            num_periods, edu_start, edu_max, min_idx)

        # PYTHON
        py_a, py_b, py_c = py_lib.create_state_space(num_periods, edu_start,
            edu_max, min_idx)

        # Ensure equivalence
        for obj in [[fort_a, py_a], [fort_b, py_b], [fort_c, py_c]]:
            np.testing.assert_allclose(obj[0], obj[1])

    for _ in range(100):

        # Draw random request for testing purposes
        num_covars = np.random.random_integers(2, 10)
        num_agents = np.random.random_integers(100, 1000)
        eps = np.random.normal(size=num_agents)
        beta = np.random.normal(size=num_covars)

        # Generate sample
        exog = np.random.sample((num_agents, num_covars))
        exog[:, 0] = 1
        endog = np.dot(exog, beta) + eps

        # Run statsmodels
        results = sm.OLS(endog, exog).fit()

        # Check parameters
        py = results.params
        f90 = fort_debug.wrapper_get_coefficients(endog, exog, num_covars,
                num_agents)
        np.testing.assert_almost_equal(py, f90)

        # Check prediction
        py = results.predict(exog)
        f90 = fort_debug.wrapper_point_predictions(exog, f90, num_agents)
        np.testing.assert_almost_equal(py, f90)

        # Check coefficient of determination
        py = results.rsquared
        f90 = fort_debug.wrapper_get_r_squared(endog, f90, num_agents)
        np.testing.assert_almost_equal(py, f90)

    # Cleanup
    cleanup()


def test_97():
    """ Compare results between FORTRAN and PYTHON of selected functions. The
    file python/f2py/debug_interface.f90 provides the F2PY bindings.
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', False)

    import robupy.python.f2py.f2py_debug as fort

    for _ in range(100):

        # Draw random requests for testing purposes.
        num_draws = np.random.random_integers(2, 1000)
        dim = np.random.random_integers(1, 6)
        mean = np.random.uniform(-0.5, 0.5, (dim))

        matrix = (np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim))
        cov = np.dot(matrix, matrix.T)

        # Singular Value Decomposition
        py = scipy.linalg.svd(matrix)
        f90 = fort.wrapper_svd(matrix, dim)

        for i in range(3):
            np.testing.assert_allclose(py[i], f90[i], rtol=1e-05, atol=1e-06)

        # Pseudo-Inverse
        py = np.linalg.pinv(matrix)
        f90 = fort.wrapper_pinv(matrix, dim)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Inverse
        py = np.linalg.inv(cov)
        f90 = fort.wrapper_inverse(cov, dim)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Determinant
        py = np.linalg.det(cov)
        f90 = fort.wrapper_determinant(cov)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Trace
        py = np.trace(cov)
        f90 = fort.wrapper_trace(cov)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Cholesky decomposition
        f90 = fort.wrapper_cholesky(cov, dim)
        py = np.linalg.cholesky(cov)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Random normal deviates. This only tests the interface, requires
        # visual inspection in IPYTHON notebook as well.
        fort.wrapper_standard_normal(num_draws)
        fort.wrapper_multivariate_normal(mean, cov, num_draws, dim)

        # Clipping values below and above bounds.
        num_values = np.random.random_integers(1, 10000)
        lower_bound = np.random.randn()
        upper_bound = lower_bound + np.random.ranf()
        values = np.random.normal(size=num_values)

        f90 = fort.wrapper_get_clipped_vector(values, lower_bound, upper_bound,
                                         num_values)
        py = np.clip(values, lower_bound, upper_bound)

        np.testing.assert_almost_equal(py, f90)

    # Cleanup
    cleanup()


def test_98():
    """  Compare results from the RESTUD program and the ROBUPY package.
    """

    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)

    # Prepare RESTUD program
    os.chdir('modules')
    os.system(' gfortran -fcheck=bounds -o dp3asim dp3asim.f95 >'
              ' /dev/null 2>&1')
    os.remove('pei_additions.mod')
    os.remove('imsl_replacements.mod')
    os.chdir('../')

    # Impose some constraints on the initialization file which ensures that
    # the problem can be solved by the RESTUD code. The code is adjusted to
    # run with zero disturbances.
    constraints = dict()
    constraints['edu'] = (10, 20)
    constraints['level'] = 0.00
    constraints['debug'] = True
    constraints['eps_zero'] = True

    version = np.random.choice(['FORTRAN', 'F2PY', 'PYTHON'])
    constraints['version'] = version

    # Generate random initialization file. The RESTUD code uses the same random
    # draws for the solution and simulation of the model. Thus, the number of
    # draws is required to be less or equal to the number of agents.
    init_dict = generate_random_dict(constraints)

    num_agents = init_dict['SIMULATION']['agents']
    num_draws = init_dict['SOLUTION']['draws']
    if num_draws < num_agents:
        init_dict['SOLUTION']['draws'] = num_agents

    print_random_dict(init_dict)

    # Indicate RESTUD code the special case of zero disturbance.
    open('.restud.testing.scratch', 'a').close()

    # Perform toolbox actions
    robupy_obj = read('test.robupy.ini')

    # This flag aligns the random components between the RESTUD program and
    # ROBUPY package. The existence of the file leads to the RESTUD program
    # to write out the random components.
    init_dict = robupy_obj.get_attr('init_dict')

    transform_robupy_to_restud(init_dict)

    # Solve model using RESTUD code.
    os.system('./modules/dp3asim > /dev/null 2>&1')

    # Solve model using ROBUPY package.
    solve(robupy_obj)

    # Compare the simulated datasets generated by the programs.
    py = pd.DataFrame(np.array(np.genfromtxt('data.robupy.dat',
            missing_values='.'), ndmin=2)[:, -4:])

    fort = pd.DataFrame(np.array(np.genfromtxt('ftest.txt',
            missing_values='.'), ndmin=2)[:, -4:])

    assert_frame_equal(py, fort)

    # Cleanup
    cleanup()


def test_99():
    """ Testing whether the results from a fast and slow execution of the
    code result in identical simulate datasets.
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)

    # Constraint to risk model
    constraints = dict()
    constraints['debug'] = True
    constraints['measure'] = 'kl'

    # We need to ensure that the number of agents is less or equal than the
    # number of draws so that we can standardize the disturbances across
    # implementations.
    num_draws = np.random.random_integers(1, 100)
    num_agents = np.random.random_integers(1, num_draws)

    constraints['agents'] = num_agents
    constraints['draws'] = num_draws

    # Just making sure that it also works for this special case. Note that
    # this special case is currently only working in the risk case.
    if np.random.choice([True, False, False, False]):
        constraints['level'] = 0.00
        constraints['eps_zero'] = True

    # Generate random initialization file.
    init_dict = generate_random_dict(constraints)

    # Ensure standardization across implementations.
    write_disturbances(init_dict)

    # Initialize containers
    base = None

    for version in ['PYTHON', 'F2PY', 'FORTRAN']:

        # This ensures that the optimized version agrees with all other
        # implementations as well.
        if version in ['OPTIMIZATION']:
            compile_package('--fortran --debug --optimization', True)
            version = 'FORTRAN'

        # Prepare initialization file
        init_dict['PROGRAM']['version'] = version

        print_random_dict(init_dict)

        # Simulate the ROBUPY package
        os.system('robupy-solve --model test.robupy.ini')

        # Load simulated data frame
        data_frame = pd.read_csv('data.robupy.dat', delim_whitespace=True)

        data_frame.to_csv(version + '.dat')

        # Compare
        if base is None:
            base = data_frame.copy()

        assert_frame_equal(base, data_frame)

    # Cleanup
    cleanup()
