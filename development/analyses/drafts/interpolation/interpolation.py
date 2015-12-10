#!/usr/bin/env python
""" I use this module to acquaint myself with the interpolation scheme
proposed in Keane & Wolpin (1994).
"""

# standard library

import numpy as np
import statsmodels.api as sm
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# project library
from robupy import *
from robupy.auxiliary import create_disturbances
from robupy.python.py.risk import get_payoffs_risk
from robupy.auxiliary import replace_missing_values
from robupy.python.py.auxiliary import get_total_value

# Solve the baseline model and get the estimated expected future values ...
robupy_obj = read('model.robupy.ini')
robupy_obj = solve(robupy_obj)


periods_emax = robupy_obj.get_attr('periods_emax')
periods_payoffs_ex_ante = robupy_obj.get_attr('periods_payoffs_ex_ante')
states_all = robupy_obj.get_attr('states_all')
num_periods = robupy_obj.get_attr('num_periods')


states_number_period = robupy_obj.get_attr('states_number_period')


num_draws = robupy_obj.get_attr('num_draws')
periods_payoffs_ex_ante = robupy_obj.get_attr('periods_payoffs_ex_ante')

delta = robupy_obj.get_attr('delta')
edu_max = robupy_obj.get_attr('edu_max')
edu_start = robupy_obj.get_attr('edu_start')
mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')



max_states_period = max(states_number_period)

periods_eps_relevant = create_disturbances(robupy_obj, False)

# Initialize containers with missing values
periods_emax_int = np.tile(-99.00, (num_periods, max_states_period))

periods_payoffs_ex_post = np.tile(-99.00, (num_periods,
                                               max_states_period, 4))
periods_future_payoffs = np.tile(-99.00, (num_periods,
                                               max_states_period, 4))

################################################################################
# Interpolation Code
################################################################################

#
#
# Points can be too small for the interpolation routine to work properly. I
# need an easy exit.
#
#
#
# Options
num_points = 1000

################################################################################
# Simulation Grid
################################################################################
from auxiliary import create_simulation_grid

simulation_grid = create_simulation_grid(num_points, max_states_period,
                           states_number_period, num_periods)

################################################################################
################################################################################

# Iterate backward through all periods
for period in range(num_periods - 1, -1, -1):

    # Auxiliary object
    num_states = states_number_period[period]
    any_interpolated = (num_points < num_states)

    eps_relevant = periods_eps_relevant[period, :, :]
    ############################################################################
    if any_interpolated:
    ############################################################################
        # Constructing an indicator whether a state will be simulated or
        # interpolated.
        is_simulated = np.tile(False, num_states)
        for i, k in enumerate(range(num_states)):
            if k in simulation_grid[period, :]:
                is_simulated[i] = True

        # Constructing the dependent variable for all states, including the ones
        # where simulation will take place.
        maxe_deviations = np.tile(np.nan, (num_states, 8))
        maxe = np.tile(np.nan, num_states)

        for i, k in enumerate(range(num_states)):

            payoffs_ex_ante = periods_payoffs_ex_ante[period, k, :]

            expected_values, _, _ = get_total_value(period, num_periods, delta,
                payoffs_ex_ante, [1.00, 1.00, 0.00, 0.00], edu_max, edu_start,
                mapping_state_idx, periods_emax, k, states_all)

            maxe[i] = max(expected_values)

            deviations = maxe[i] - expected_values

            maxe_deviations[i, :] = np.hstack((deviations, np.sqrt(deviations)))

        emax_deviation = np.tile(np.nan, num_states)

        for i, k in enumerate(range(num_states)):
            # Skip over points that will be interpolated.
            if not is_simulated[i]:
                continue

            # Extract payoffs
            payoffs_ex_ante = periods_payoffs_ex_ante[period, k, :]

            # Simulate the expected future value.
            emax, _, _ = \
                get_payoffs_risk(num_draws, eps_relevant, period, k,
                    payoffs_ex_ante, edu_max, edu_start, mapping_state_idx,
                    states_all, num_periods, periods_emax_int, delta)

            # Collect
            emax_deviation[i] = emax - maxe[i]

        # Create prediction model
        model = sm.OLS(emax_deviation[is_simulated], sm.add_constant(
            maxe_deviations[is_simulated]))

        results = model.fit()

        predictions = results.predict(sm.add_constant(maxe_deviations)) + maxe
        # Replace the simulated values with their actual realization instead of
        # just the predictions.
        predictions[is_simulated] = emax_deviation[is_simulated] + maxe[is_simulated]

        periods_emax_int[period, :num_states] = predictions

        # Checks
        assert (model.nobs == min(num_points, num_states))
        assert (results.params.shape == (9,))
        assert (np.all(np.isfinite(results.params)))
    ############################################################################
    else:
    ############################################################################
        # Is there a nice way to simply check whether there were any problems in
        # the OLS regression?
        # Loop over all possible states
        for k in range(num_states):

            # Extract payoffs
            payoffs_ex_ante = periods_payoffs_ex_ante[period, k, :]

            # Simulate the expected future value.
            emax, payoffs_ex_post, future_payoffs = \
                get_payoffs_risk(num_draws, eps_relevant, period, k,
                    payoffs_ex_ante, edu_max, edu_start, mapping_state_idx,
                    states_all, num_periods, periods_emax_int, delta)

            # Collect information
            periods_payoffs_ex_post[period, k, :] = payoffs_ex_post
            periods_future_payoffs[period, k, :] = future_payoffs

            # Collect
            periods_emax_int[period, k] = emax

# TODO: Test case in absence of randomness, develop this script in a test
# case later, testing if interpolation code works for random requests,
# results identical if NUM_POINTS is number of maximum states, add a lot of
# checks for the interpolation matrices (if any). Write out information about
# interpolation model to the log file (R squared).
replace_missing_values(periods_emax_int)
np.testing.assert_array_almost_equal(periods_emax, periods_emax_int)
