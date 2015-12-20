#!/usr/bin/env python
""" This module creates a graphical illustration of all admissible values.
The idea is to use this to split the discussion between all admissible values
and really making a decision.
"""

# standard library
from scipy.optimize import minimize
import numpy as np
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# project library
from robupy import *

from robupy.python.py.auxiliary import simulate_emax

from robupy.python.py.ambiguity import transform_disturbances_ambiguity
from robupy.python.py.ambiguity import _prep_kl

from robupy.tests.random_init import print_random_dict
from robupy.auxiliary import create_disturbances

# Make sure that fast evaluation of the model is possible
from modules.auxiliary import compile_package
compile_package('--fortran --optimization', True)
################################################################################
################################################################################
# This is a slightly modified copy of the criterion function in the ambiguity
# module. The ability to switch the sign was added to allow for maximization
# as well as minimization.


def _criterion(x, num_draws, eps_relevant, period, k, payoffs_systematic, edu_max,
        edu_start, mapping_state_idx, states_all, num_periods, periods_emax,
        delta, sign=1):
    """ Simulate expected future value for alternative shock distributions.
    """
    # Checks
    assert (sign in [1, -1])

    # Transformation of standard normal deviates to relevant distributions.
    eps_relevant_emax = transform_disturbances_ambiguity(eps_relevant, x)

    # Simulate the expected future value for a given parametrization.
    simulated, _, _ = simulate_emax(num_periods, num_draws, period, k,
                        eps_relevant_emax, payoffs_systematic, edu_max, edu_start,
                        periods_emax, states_all, mapping_state_idx, delta)

    # Finishing
    return sign*simulated
################################################################################
################################################################################
AMBIGUITY_GRID = [0.00, 0.01, 0.02, 0.03]

# Initialize auxiliary objects
period, rslt = 1, dict()
# Iterate over the alternative levels of ambiguity.
for ambi in AMBIGUITY_GRID:
    # Initialize container
    rslt[ambi] = []
    # Solve the model for alternative values of ambiguity.
    robupy_obj = read('model.robupy.ini')
    init_dict = robupy_obj.get_attr('init_dict')
    init_dict['AMBIGUITY']['level'] = ambi
    print_random_dict(init_dict)
    robupy_obj = read('test.robupy.ini')
    robupy_obj = solve(robupy_obj)
    # Distribute objects from solution. These are then used to reproduce the
    # EMAX calculations from inside the toolbox.
    periods_payoffs_systematic = robupy_obj.get_attr('periods_payoffs_systematic')
    mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')
    periods_emax = robupy_obj.get_attr('periods_emax')
    is_ambiguous = robupy_obj.get_attr('is_ambiguous')
    num_periods = robupy_obj.get_attr('num_periods')
    states_all = robupy_obj.get_attr('states_all')
    num_draws = robupy_obj.get_attr('num_draws')
    edu_start = robupy_obj.get_attr('edu_start')
    edu_max = robupy_obj.get_attr('edu_max')
    shocks = robupy_obj.get_attr('shocks')
    delta = robupy_obj.get_attr('delta')
    level = robupy_obj.get_attr('level')
    # Iterate over the four alternative decisions for an agent at the
    # beginning of a decision tree.
    for state_indicator in range(4):
        # Determine the maximum and minimum value of the value function.
        bounds = []
        for sign in [1, -1]:
            # Extract subsets of information
            payoffs_systematic = periods_payoffs_systematic[period, state_indicator, :]
            periods_eps_relevant = create_disturbances(robupy_obj, False)
            eps_relevant = periods_eps_relevant[period, :, :]
            # In the risk-only case, no modification of the distribution of
            # the disturbances is required. This case is only handled in this
            # script for benchmarking purposes.
            eps_relevant_emax = eps_relevant
            # In the case of ambiguity, the upper and lower bounds for the
            # value function are required.
            if is_ambiguous:
                # Set up the arguments for the optimization request.
                args = (num_draws, eps_relevant, period, state_indicator,
                        payoffs_systematic, edu_max, edu_start, mapping_state_idx,
                        states_all, num_periods, periods_emax, delta, sign)
                constraints = _prep_kl(shocks, level)
                x0, options = [0.0, 0.0], dict()
                options['maxiter'] = 100000000
                # Run the optimization to determine upper and lower bound
                # depending on the sign variable.
                opt = minimize(_criterion, x0, args, method='SLSQP', options=options,
                        constraints=constraints)
                # Check success
                assert (opt['success'] == True)
                # Transform relevant disturbances according to results from
                # optimization step.
                eps_relevant_emax = transform_disturbances_ambiguity(eps_relevant, opt['x'])
            # Back out the relevant EMAX using the results from the
            # optimization.
            emax_simulated, _, _ = simulate_emax(num_periods, num_draws, period,
                state_indicator, eps_relevant_emax, payoffs_systematic, edu_max,
                edu_start, periods_emax, states_all, mapping_state_idx, delta)
            # Collect results in tuples for each level of ambiguity.
            bounds += [emax_simulated]
            # This is a manual test as the results should be identical for
            # risk-only case and the worst-case ambiguity evaluation. They
            # will be different if the best-case evaluation is requested.
            if sign == 1 or ambi == 0.0:
                np.testing.assert_almost_equal(emax_simulated,
                    periods_emax[period, state_indicator])
        # Collect bounds
        rslt[ambi] += [bounds]
# Let us run some basic checks on the results object. This is also useful to
# understand the structure of the object when revisiting the code later.
assert (set(rslt.keys()) == set(AMBIGUITY_GRID))
for ambi in AMBIGUITY_GRID:
    assert (len(rslt[ambi]) == 4)
    for bounds in rslt[ambi]:
        if ambi == 0.00:
            assert (bounds[0] == bounds[1])
        else:
            assert (bounds[0] < bounds[1])
################################################################################
# Further Processing
################################################################################
for ambi in AMBIGUITY_GRID:
    print(ambi, rslt[ambi])