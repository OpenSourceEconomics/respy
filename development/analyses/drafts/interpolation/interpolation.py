#!/usr/bin/env python
""" I use this module to acquaint myself with the interpolation scheme
proposed in Keane & Wolpin (1994).
"""

# standard library
import statsmodels.api as sm
import numpy as np

import random
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# project library
from robupy.python.py.auxiliary import get_total_value
from robupy.auxiliary import replace_missing_values
from robupy.python.py.risk import get_payoffs_risk
from robupy.auxiliary import create_disturbances
from modules.auxiliary import compile_package

from robupy.tests.random_init import print_random_dict
from robupy.tests.random_init import generate_random_dict


from robupy import *


from auxiliary import create_simulation_grid


def interpolation():

    # Solve the baseline model and get the estimated expected future values ...
    robupy_obj = read('test.robupy.ini')
    robupy_obj = solve(robupy_obj)

    periods_emax = robupy_obj.get_attr('periods_emax')
    states_all = robupy_obj.get_attr('states_all')
    num_periods = robupy_obj.get_attr('num_periods')


    states_number_period = robupy_obj.get_attr('states_number_period')


    num_draws = robupy_obj.get_attr('num_draws')

    shocks = robupy_obj.get_attr('shocks')
    shifts = (shocks[0, 0]/2.0, shocks[1, 1]/2.0)


    periods_payoffs_systematic = robupy_obj.get_attr('periods_payoffs_systematic')

    delta = robupy_obj.get_attr('delta')
    edu_max = robupy_obj.get_attr('edu_max')
    edu_start = robupy_obj.get_attr('edu_start')
    mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')

    max_states_period = max(states_number_period)

    periods_eps_relevant = create_disturbances(robupy_obj, False)

    # Initialize containers with missing values
    periods_emax_int = np.tile(-99.00, (num_periods, max_states_period))

    # Random sample of simulation points. There is a special case, when the
    # number of simulation points is exactly equal to the maximum number of
    # states in any period. In this case the interpolation code is run and
    # should result in identical EMAX numbers. This is true even in the case
    # of randomness.
    num_points = max(100, np.random.choice(max_states_period))
    num_points = np.random.choice([max_states_period, num_points], p=[0.8, 0.2])
    ############################################################################
    ############################################################################

    # Iterate backward through all periods
    for period in range(num_periods - 1, -1, -1):

        print('\n', period)

        # Auxiliary object
        eps_relevant = periods_eps_relevant[period, :, :]
        num_states = states_number_period[period]

        # TODO: This will be more flexible once integrated in the
        # initialization file. The interpolation routines are used in case of
        # equality for testing purposes only.
        any_interpolated = (num_points <= num_states)

        ############################################################################
        if any_interpolated:
        ############################################################################
            # Drawing random interpolation points
            interpolation_points = np.random.choice(range(num_states),
                                        size=num_points, replace=False)

            # Constructing an indicator whether a state will be simulated or
            # interpolated.
            is_simulated = np.tile(False, num_states)
            is_simulated[interpolation_points] = True

            # Constructing the dependent variable for all states, including the
            # ones where simulation will take place. All information will be
            # used in either the construction of the prediction model or the
            # prediction step.
            independent_variables = np.tile(np.nan, (num_states, 9))
            maxe = np.tile(np.nan, num_states)

            for i, k in enumerate(range(num_states)):

                payoffs_systematic = periods_payoffs_systematic[period, k, :]

                # TODO: This is not MAXE!
                expected_values, _, _ = get_total_value(period, num_periods,
                    delta, payoffs_systematic, [1.00, 1.00, 0.00, 0.00],
                    edu_max, edu_start, mapping_state_idx, periods_emax, k,
                    states_all)

                maxe[i] = max(expected_values)

                deviations = maxe[i] - expected_values

                independent_variables[i, :8] = np.hstack((deviations,
                                                    np.sqrt(deviations)))

            # Add intercept to set of independent variables and replace
            # infinite values.
            independent_variables[:, 8] = 1

            # TODO: Is this the best we can do? What are the exact consequences?
            independent_variables[np.isinf(independent_variables)] = \
                -50000

            # Constructing the dependent variables for at the random subset of
            # points where the EMAX is actually calculated.
            dependent_variable = np.tile(np.nan, num_states)
            for i, k in enumerate(range(num_states)):
                # Skip over points that will be interpolated.
                if not is_simulated[i]:
                    continue
                # Extract payoffs
                payoffs_systematic = periods_payoffs_systematic[period, k, :]
                # Simulate the expected future value.
                emax, _, _ = get_payoffs_risk(num_draws, eps_relevant, period, k,
                                payoffs_systematic, edu_max, edu_start,
                                mapping_state_idx, states_all, num_periods,
                                periods_emax_int, delta)
                # Construct dependent variable
                dependent_variable[i] = emax - maxe[i]

            # Create prediction model based on the random subset of points where
            # the EMAX is actually simulated and thus dependent and
            # independent variables are available.
            model = sm.OLS(dependent_variable[is_simulated],
                        independent_variables[is_simulated])

            results = model.fit()

            # Use the model to predict EMAX for all states and subsequently
            # replace the values where actual values are available. As in
            # Keane & Wolpin (1994), negative predictions are truncated to zero.
            predictions_diff = results.predict(independent_variables)
            predictions_diff = np.clip(predictions_diff, 0.00, None)

            # Construct predicted EMAX for all states and the replace
            # interpolation points with simulated values.
            predictions = predictions_diff + maxe
            predictions[is_simulated] = dependent_variable[is_simulated] + \
                                            maxe[is_simulated]

            # Store results
            periods_emax_int[period, :num_states] = predictions

            # Checks
            assert (np.all(predictions_diff >= 0.00))
            assert (model.nobs == min(num_points, num_states))
            assert (results.params.shape == (9,))
            assert (np.all(np.isfinite(results.params)))
        #############################################################################
        else:
        #############################################################################
            for k in range(num_states):

                # Extract payoffs
                payoffs_systematic = periods_payoffs_systematic[period, k, :]

                # Simulate the expected future value.
                emax, _, _ = \
                    get_payoffs_risk(num_draws, eps_relevant, period, k,
                        payoffs_systematic, edu_max, edu_start, mapping_state_idx,
                        states_all, num_periods, periods_emax_int, delta)

                # Collect
                periods_emax_int[period, k] = emax

    # TODO:Add a lot of checks for the interpolation matrices (if any).
    # Write out information about interpolation model to the log file (R
    # squared).
    shocks = robupy_obj.get_attr('shocks')
    if np.all(shocks == 0.00) or (max_states_period == num_points):
        print('TESTING')
        replace_missing_values(periods_emax_int)
        np.testing.assert_array_almost_equal(periods_emax, periods_emax_int)

# The test case it that the results in the absence of noise should be identical
# regardless of all other parameters of the model.
# Ensure that fast solution methods are available
# TODO: Comment back in ...
#compile_package('--fortran --debug', True)

# Constraint to risk only model without any randomness then the
random.seed(12345)
for count, _ in enumerate(range(10000)):

    seed = random.randrange(0, 10000)

    print(count, seed, '\n')
    np.random.seed(seed)

    constraints = dict()
    constraints['periods'] = np.random.random_integers(10, 15)
    # Additional testing in the absence of randomness, where the MAXE and
    # EMAX are identical
    constraints['eps_zero'] = np.random.choice([True, False])

    # TODO: These restrictions have to be loosened in due time
    constraints['version'] = 'PYTHON'
    constraints['level'] = 0.00

    # Sample a random estimation request and write it to disk.
    init_dict = generate_random_dict(constraints)

    print_random_dict(init_dict)

    # Run interpolation routine
    interpolation()



