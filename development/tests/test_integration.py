""" This modules contains some additional tests that are only used in long-run
development tests.
"""

# standard library
from pandas.util.testing import assert_frame_equal

from scipy.optimize.slsqp import _minimize_slsqp
from scipy.optimize import approx_fprime
from scipy.optimize import rosen_der
from scipy.optimize import rosen
from scipy.stats import norm

import statsmodels.api as sm
import pandas as pd
import numpy as np

import scipy
import sys
import os

# testing library
from material.auxiliary import transform_robupy_to_restud
from material.auxiliary import write_interpolation_grid
from material.auxiliary import write_disturbances
from material.auxiliary import build_f2py_testing
from material.auxiliary import compile_package
from material.auxiliary import cleanup, distribute_model_description

# ROBUPY import
sys.path.insert(0, os.environ['ROBUPY'])
from robupy import *

from robupy.python.py.python_library import _get_simulated_indicator
from robupy.python.py.python_library import _get_exogenous_variables
from robupy.python.py.python_library import _get_endogenous_variable
from robupy.python.py.python_library import _get_predictions
from robupy.python.py.python_library import get_payoffs

from robupy.python.solve_python import solve_python_bare

from robupy.tests.random_init import generate_random_dict
from robupy.tests.random_init import print_random_dict
from robupy.tests.random_init import generate_init

from robupy.python.py.ambiguity import get_payoffs_ambiguity
from robupy.python.py.auxiliary import simulate_emax
from robupy.python.py.ambiguity import _divergence
from robupy.python.py.ambiguity import _criterion

from robupy.auxiliary import distribute_model_paras
from robupy.auxiliary import replace_missing_values
from robupy.auxiliary import create_disturbances





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
    constraints['is_deterministic'] = True

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

        # Extract class attributes
        states_number_period, periods_emax = \
            distribute_model_description(robupy_obj,
                'states_number_period', 'periods_emax')

        # Store and check results
        if baseline is None:
            baseline = periods_emax
        else:
            np.testing.assert_array_almost_equal(baseline, periods_emax)

        # Updates for second iteration. This ensures that there is at least
        # one interpolation taking place.
        init_dict['INTERPOLATION']['points'] = max(states_number_period) - 1
        init_dict['INTERPOLATION']['apply'] = True

    # Cleanup
    cleanup()


def test_98():
    """ Testing whether the results from a fast and slow execution of the
    code result in identical simulate datasets.
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)

    # Constraint to risk model
    max_draws = np.random.random_integers(1, 100)

    constraints = dict()
    constraints['max_draws'] = max_draws
    constraints['measure'] = 'kl'

    # Just making sure that it also works for this special case. Note that
    # this special case is currently only working in the risk case.
    if np.random.choice([True, False, False, False]):
        constraints['level'] = 0.00
        constraints['is_deterministic'] = True

    # Generate random initialization file.
    init_dict = generate_random_dict(constraints)

    # Align randomness across implementations
    num_periods = init_dict['BASICS']['periods']
    write_disturbances(num_periods, max_draws)

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


''' Main
'''


def test_99():
    """ Testing whether random datasets can be simulated and processed.
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)

    for i in range(5):

        # Generate random initialization file
        generate_init()

        robupy_obj = read('test.robupy.ini')

        solve(robupy_obj)

        simulate(robupy_obj)

        process(robupy_obj)

    # Cleanup
    cleanup()


def test_101():
    """ Testing the equality of an evaluation of the criterion function for
    a random request.
    """
    # Ensure that fast solution methods are available
    compile_package('--fortran --debug', True)

    # Run evaluation for multiple random requests.
    for _ in range(5):

        # Constraints
        is_deterministic = np.random.choice([True, False], p=[0.10, 0.9])
        # TODO: There is another case with special attention, Interpolation
        is_myopic = np.random.choice([True, False], p=[0.10, 0.9])
        max_draws = np.random.random_integers(10, 100)

        constraints = dict()
        constraints['is_deterministic'] = is_deterministic
        constraints['is_myopic'] = is_myopic
        constraints['max_draws'] = max_draws

        # Generate random initialization file
        init_dict = generate_init(constraints)

        # Write out random components and interpolation grid to align the three
        # implementations.
        num_periods = init_dict['BASICS']['periods']
        write_disturbances(num_periods, max_draws)
        write_interpolation_grid('test.robupy.ini')

        # Clean evaluations based on interpolation grid,
        base = None

        for version in ['PYTHON', 'F2PY', 'FORTRAN']:

            robupy_obj = read('test.robupy.ini')

            robupy_obj = solve(robupy_obj)

            data_frame = simulate(robupy_obj)

            robupy_obj.unlock()

            robupy_obj.set_attr('version',  version)

            robupy_obj.lock()

            robupy_obj, eval_ = evaluate(robupy_obj, data_frame)

            if base is None:
                base = eval_

            np.testing.assert_allclose(base, eval_, rtol=1e-05, atol=1e-06)

            # We know even more for the deterministic case.
            if constraints['is_deterministic']:
                assert (eval_ in [0.0, 1.0])

        cleanup()
