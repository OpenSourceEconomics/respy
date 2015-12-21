#!/usr/bin/env python
""" This module compares the results from a full and interpolated solution to
the basic model. It reproduces the first column of table 2.1 in Keane &
Wolpin (1994).
"""

# standard library
import numpy as np

import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# project library
from robupy.python.py.auxiliary import get_total_value
from robupy.auxiliary import create_disturbances

from robupy.tests.random_init import print_random_dict

from robupy.tests.random_init import print_random_dict
from robupy import *


def assess_approximation(num_agents, states_all, num_periods, mapping_state_idx,
        periods_payoffs_systematic, periods_eps_relevant, edu_max, edu_start,
        delta, periods_emax_true, periods_emax_inter):
    """ Sample simulation
    """
    # Initialize results dictionary
    failure_count_information = dict()
    for period in range(num_periods):
        failure_count_information[period] = 0

    # Iterate over numerous agents for all periods.
    for i in range(num_agents):

        current_state = states_all[0, 0, :].copy()

        # Iterate over each period for the agent
        for period in range(num_periods):

            # Distribute state space
            exp_A, exp_B, edu, edu_lagged = current_state

            k = mapping_state_idx[period, exp_A, exp_B, edu, edu_lagged]

            # Select relevant subset
            payoffs_systematic = periods_payoffs_systematic[period, k, :]
            disturbances = periods_eps_relevant[period, i, :]

            # Get total value of admissible states using full solution
            total_payoffs_full, _, _ = get_total_value(period, num_periods,
                delta, payoffs_systematic, disturbances, edu_max, edu_start,
                mapping_state_idx, periods_emax_true, k, states_all)

            # Get total value of admissible states using interpolated solution
            total_payoffs_interpolated, _, _ = get_total_value(period,
                num_periods, delta, payoffs_systematic, disturbances, edu_max,
                edu_start, mapping_state_idx, periods_emax_inter, k, states_all)

            # Compare the implications for choice and assess potential failure
            is_failure = np.argmax(total_payoffs_full) != np.argmax(
                total_payoffs_interpolated)
            if is_failure:
                failure_count_information[period] += 1

            # Determine optimal choice based on full solution
            max_idx = np.argmax(total_payoffs_full)

            # Update work experiences and education
            if max_idx == 0:
                current_state[0] += 1
            elif max_idx == 1:
                current_state[1] += 1
            elif max_idx == 2:
                current_state[2] += 1

            # Update lagged education
            current_state[3] = 0

            if max_idx == 2:
                current_state[3] = 1

    # Create total failure count and add additional information.
    failure_count_information['num_agents'] = num_agents
    failure_count_information['num_periods'] = num_periods
    failure_count_information['total'] = 0
    for period in range(period):
        failure_count_information['total'] += \
            failure_count_information[period]

    # Summary measure of correct prediction.
    return failure_count_information

################################################################################
################################################################################
# Read in baseline initialization file
robupy_obj = read('model.robupy.ini')
init_dict = robupy_obj.get_attr('init_dict')

# Run the toolbox with and without interpolation
baseline = None

for is_true in [True, False]:

    if is_true:
        # EMAX Integration
        init_dict['SOLUTION']['draws'] = 100000
        # Value Function Interpolation
        init_dict['INTERPOLATION']['apply'] = False
    else:
        # EMAX Integration
        init_dict['SOLUTION']['draws'] = 2000
        # Value Function Interpolation
        init_dict['INTERPOLATION']['apply'] = True
        init_dict['INTERPOLATION']['points'] = 2000

    print_random_dict(init_dict)

    robupy_obj = read('test.robupy.ini')

    robupy_obj = solve(robupy_obj)

    periods_emax = robupy_obj.get_attr('periods_emax')

    if is_true:
        periods_emax_true = periods_emax
    else:
        periods_emax_inter = periods_emax

periods_eps_relevant = create_disturbances(robupy_obj, False)

# How to assess the quality of the approximation?
periods_payoffs_systematic = robupy_obj.get_attr('periods_payoffs_systematic')

mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')

periods_emax = robupy_obj.get_attr('periods_emax')

num_periods = robupy_obj.get_attr('num_periods')

num_agents = robupy_obj.get_attr('num_agents')

states_all = robupy_obj.get_attr('states_all')

edu_start = robupy_obj.get_attr('edu_start')

edu_max = robupy_obj.get_attr('edu_max')

delta = robupy_obj.get_attr('delta')

failure_count_information = assess_approximation(num_agents, states_all,
        num_periods, mapping_state_idx, periods_payoffs_systematic,
        periods_eps_relevant, edu_max, edu_start, delta, periods_emax_true,
        periods_emax_inter)

print(failure_count_information)


