#!/usr/bin/env python
""" This module creates a graphical illustration of all admissible values.
The idea is to use this to split the discussion between all admissible values
and really making a decision.
"""

# standard library
from multiprocessing import Pool
from functools import partial
from scipy.optimize import minimize

import numpy as np

import argparse
import sys
import os

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
sys.path.insert(0, os.environ['ROBUPY'])

# project library
from robupy import *
from robupy.python.py.auxiliary import simulate_emax
from robupy.python.py.ambiguity import transform_disturbances_ambiguity
from robupy.python.py.ambiguity import _prep_kl, _divergence

from robupy.auxiliary import create_disturbances


################################################################################
################################################################################
# Specify request
period, k = 0, 0

# Solve baseline distribution
robupy_obj = read('model.robupy.ini')
robupy_obj = solve(robupy_obj)

# Distribute objects from solution
periods_payoffs_ex_ante = robupy_obj.get_attr('periods_payoffs_ex_ante')
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

# Extract subsets of information
payoffs_ex_ante = periods_payoffs_ex_ante[period, k, :]


# TODO: Would be great to use the packaged version instead.
def _criterion(x, num_draws, eps_relevant, period, k, payoffs_ex_ante, edu_max,
        edu_start, mapping_state_idx, states_all, num_periods, periods_emax,
        delta):
    """ Simulate expected future value for alternative shock distributions.
    """

    # Transformation of standard normal deviates to relevant distributions.
    eps_relevant_emax = transform_disturbances_ambiguity(eps_relevant, x)

    # Simulate the expected future value for a given parametrization.
    simulated, _, _ = simulate_emax(num_periods, num_draws, period, k,
                        eps_relevant_emax, payoffs_ex_ante, edu_max, edu_start,
                        periods_emax, states_all, mapping_state_idx, delta)

    # Finishing
    rslt = simulated
    print(rslt, x, _divergence(x, shocks, level))
    return rslt



# This is part of the code is different in the case with and without
# ambiguity. Note that the expected future values will be different as one is
# based on the worst case evaluation. This needs to be reproduced next.
# TODO: INTEGRATE WORST CASE EVALUATION, THEN TWEAK IT FOR BEST CASE AS WELL.
periods_eps_relevant = create_disturbances(robupy_obj, False)
eps_relevant = periods_eps_relevant[period, :, :]

if not is_ambiguous:
    eps_relevant_emax = eps_relevant
else:

    args = (num_draws, eps_relevant, period, k, payoffs_ex_ante, edu_max,
                edu_start, mapping_state_idx, states_all, num_periods, periods_emax,
                delta)

    # Initialize options.
    import random
    options = dict()
    options['maxiter'] = 10000000

    x0 = np.array([random.random(),  random.random()])*0.000001

    constraints = _prep_kl(shocks, level)

    opt = minimize(_criterion, x0, args, method='SLSQP', options=options,
                           constraints=constraints)

    print(opt['success'])

    #assert (opt['success'] is True)
    eps_relevant_emax = transform_disturbances_ambiguity(eps_relevant, opt['x'])



rslt = simulate_emax(num_periods, num_draws, period, k, eps_relevant_emax,
        payoffs_ex_ante, edu_max, edu_start, periods_emax, states_all,
        mapping_state_idx, delta)

emax_simulated, payoffs_ex_post, future_payoffs = rslt

print(emax_simulated, periods_emax[period, k])

assert (emax_simulated == periods_emax[period, k])
