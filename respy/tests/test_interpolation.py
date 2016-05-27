# standard library
import numpy as np

import pytest
import os

# testing library
from codes.auxiliary import write_interpolation_grid
from codes.random_init import generate_random_dict
from codes.random_init import generate_init

# project library
from respy.python.solve.solve_auxiliary import get_simulated_indicator
from respy.python.solve.solve_auxiliary import get_exogenous_variables
from respy.python.solve.solve_auxiliary import get_endogenous_variable
from respy.python.solve.solve_auxiliary import logging_solution
from respy.python.solve.solve_auxiliary import get_predictions

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.shared.shared_auxiliary import print_init_dict
from respy.python.shared.shared_auxiliary import create_draws

import respy.fortran.f2py_debug as fort_debug

from respy.solve import solve
from respy import RespyCls


@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ This is the special case where the EMAX better be equal to the MAXE.
        """
        # Set initial constraints
        constraints = dict()
        constraints['apply'] = False
        constraints['periods'] = np.random.randint(3, 6)
        constraints['is_deterministic'] = True

        # Initialize request
        init_dict = generate_random_dict(constraints)
        baseline = None

        # Solve with and without interpolation code
        for _ in range(2):

            # Write out request
            print_init_dict(init_dict)

            # Process and solve
            respy_obj = RespyCls('test.respy.ini')
            respy_obj = solve(respy_obj)

            # Extract class attributes
            states_number_period, periods_emax = \
                dist_class_attributes(respy_obj,
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

    def test_2(self):
        """ Further tests for the interpolation routines.
        """
        # Generate random initialization file
        generate_init()

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')
        respy_obj = solve(respy_obj)

        # Extract class attributes
        periods_payoffs_systematic, states_number_period, \
            mapping_state_idx, seed_prob, periods_emax, model_paras, \
            num_periods, states_all, num_points_interp, edu_start, num_draws_emax, \
            is_debug, edu_max, delta = \
                dist_class_attributes(respy_obj,
                    'periods_payoffs_systematic', 'states_number_period',
                    'mapping_state_idx', 'seed_prob', 'periods_emax',
                    'model_paras', 'num_periods', 'states_all', 'num_points_interp',
                    'edu_start', 'num_draws_emax', 'is_debug', 'edu_max',
                    'delta')

        # Auxiliary objects
        shocks_cholesky = dist_model_paras(model_paras, is_debug)[-1]

        # Add some additional objects required for the interfaces to the
        # functions.
        period = np.random.choice(range(num_periods))

        periods_draws_emax = create_draws(num_periods, num_draws_emax,
            seed_prob, is_debug)

        draws_emax = periods_draws_emax[period, :, :]

        num_states = states_number_period[period]

        shifts = np.random.randn(4)

        # Slight modification of request which assures that the
        # interpolation code is working.
        num_points_interp = min(num_points_interp, num_states)

        # Get the IS_SIMULATED indicator for the subset of points which are
        # used for the predication model.
        args = (num_points_interp, num_states, period, is_debug)
        is_simulated = get_simulated_indicator(*args)

        # Construct the exogenous variables for all points of the state
        # space.
        args = (period, num_periods, num_states, delta,
                periods_payoffs_systematic, shifts, edu_max, edu_start,
                mapping_state_idx, periods_emax, states_all)

        py = get_exogenous_variables(*args)
        f90 = fort_debug.wrapper_get_exogenous_variables(*args)

        np.testing.assert_equal(py, f90)

        # Distribute validated results for further functions.
        exogenous, maxe = py

        # Construct endogenous variable so that the prediction model can be
        # fitted.
        args = (period, num_periods, num_states, delta,
            periods_payoffs_systematic, edu_max, edu_start, mapping_state_idx,
            periods_emax, states_all, is_simulated, num_draws_emax, maxe,
            draws_emax, shocks_cholesky)

        py = get_endogenous_variable(*args)
        f90 = fort_debug.wrapper_get_endogenous_variable(*args)

        np.testing.assert_equal(py, replace_missing_values(f90))

        # Distribute validated results for further functions.
        endogenous = py

        # Get predictions for expected future values. We need to start the
        # logging to set up the handler for logging the output from the
        # prediction model.
        logging_solution('start')

        args = (endogenous, exogenous, maxe, is_simulated, num_points_interp,
            num_states, is_debug)

        py = get_predictions(*args)
        f90 = fort_debug.wrapper_get_predictions(*args[:-1])

        np.testing.assert_array_almost_equal(py, f90)

        logging_solution('stop')

    def test_3(self):
        """ This is a special test for auxiliary functions related to the
        interpolation setup.
        """
        # Impose constraints
        constr = dict()
        constr['periods'] = np.random.randint(2, 5)

        # Construct a random initialization file
        generate_init(constr)

        # Extract required information
        respy_obj = RespyCls('test.respy.ini')

        # Extract class attributes
        is_debug, num_periods = dist_class_attributes(respy_obj,
                'is_debug', 'num_periods')

        # Write out a grid for the interpolation
        max_states_period = write_interpolation_grid('test.respy.ini')

        # Draw random request for testing
        num_states = np.random.randint(1, max_states_period)
        candidates = list(range(num_states))

        period = np.random.randint(1, num_periods)
        num_points_interp = np.random.randint(1, num_states + 1)

        # Check function for random choice and make sure that there are no
        # duplicates.
        f90 = fort_debug.wrapper_random_choice(candidates, num_states, num_points_interp)
        np.testing.assert_equal(len(set(f90)), len(f90))
        np.testing.assert_equal(len(f90), num_points_interp)

        # Check the standard cases of the function.
        args = (num_points_interp, num_states, period, is_debug, num_periods)
        f90 = fort_debug.wrapper_get_simulated_indicator(*args)

        np.testing.assert_equal(len(f90), num_states)
        np.testing.assert_equal(np.all(f90) in [0, 1], True)

        # Test the standardization across PYTHON, F2PY, and FORTRAN
        # implementations. This is possible as we write out an interpolation
        # grid to disk which is used for both functions.
        base_args = (num_points_interp, num_states, period, is_debug)
        args = base_args
        py = get_simulated_indicator(*args)
        args = base_args + (num_periods, )
        f90 = fort_debug.wrapper_get_simulated_indicator(*args)
        np.testing.assert_array_equal(f90, 1*py)
        os.unlink('interpolation.txt')

        # Special case where number of interpolation points are same as the
        # number of candidates. In that case the returned indicator
        # should be all TRUE.
        args = (num_states, num_states, period, True, num_periods)
        f90 = fort_debug.wrapper_get_simulated_indicator(*args)
        np.testing.assert_equal(sum(f90), num_states)

    def test_4(self):
        """ This test compares the results from a solution using the
        interpolation code for the special case where the number of interpolation
        points is exactly the number of states in the final period. In this case
        the interpolation code is run and then all predicted values replaced
        with their actual values.
        """
        # Set initial constraints
        constraints = dict()
        constraints['apply'] = False
        constraints['periods'] = np.random.randint(3, 6)

        # Initialize request
        init_dict = generate_random_dict(constraints)
        baseline = None

        # Solve with and without interpolation code
        for _ in range(2):

            # Write out request
            print_init_dict(init_dict)

            # Process and solve
            respy_obj = RespyCls('test.respy.ini')
            respy_obj = solve(respy_obj)

            # Extract class attributes
            states_number_period, periods_emax = \
                dist_class_attributes(respy_obj,
                    'states_number_period', 'periods_emax')

            # Store and check results
            if baseline is None:
                baseline = periods_emax
            else:
                np.testing.assert_array_almost_equal(baseline, periods_emax)

            # Updates for second iteration
            init_dict['INTERPOLATION']['points'] = max(states_number_period)
            init_dict['INTERPOLATION']['apply'] = True

