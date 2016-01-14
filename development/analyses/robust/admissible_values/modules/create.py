#!/usr/bin/env python
""" This module creates a graphical illustration of all admissible values.
The idea is to show an agent that is in the second to last period and
contemplating the next move. This is the last period where the future payoffs
are potentially uncertain due to the ambiguity of the economic environment.
If ambiguity is present, I run two optimization problems. First, as usual,
I determine the worst-case outcome. Then I also calculate the best case
outcome.


I analyse the second to last period's choice problem as in this case there is
no accumulation of worst-cse evaluations from later periods. The set of
admissible values is always a superset.
"""

# standard library
from multiprocessing import Pool
from functools import partial

import argparse
import shutil
import socket
import sys
import os

# scipy libraries
from scipy.optimize import minimize

import pickle as pkl
import numpy as np

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']
SPEC_DIR = ROBUPY_DIR + '/development/analyses/restud/specifications'

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR + '/development/analyses/robust/_scripts')
sys.path.insert(0, ROBUPY_DIR + '/development/tests/random')
sys.path.insert(0, ROBUPY_DIR)

# _scripts
from _auxiliary import float_to_string
from _auxiliary import get_robupy_obj

# robupy library
from robupy.python.py.ambiguity import transform_disturbances_ambiguity
from robupy.python.py.auxiliary import simulate_emax
from robupy.python.py.ambiguity import _prep_kl
from robupy.tests.random_init import print_random_dict
from robupy.auxiliary import create_disturbances

from robupy import solve
from robupy import read

# local library
from auxiliary import distribute_arguments
from auxiliary import criterion

# testing library
from modules.auxiliary import compile_package

''' Main function
'''


def run(init_dict, is_debug, ambiguity_level):

    # Switch to subdirectory to store results.
    os.chdir('rslts')

    # Auxiliary objects
    name, rslt = float_to_string(ambiguity_level), []

    # Prepare directory structure
    os.mkdir(name), os.chdir(name)

    # Initialize auxiliary objects
    init_dict['AMBIGUITY']['level'] = ambiguity_level

    # Restrict number of periods for debugging purposes.
    if is_debug:
        init_dict['BASICS']['periods'] = 3
        init_dict['PROGRAM']['version'] = 'PYTHON'

    # Print initialization file for debugging purposes.
    print_random_dict(init_dict)

    # Solve the basic economy
    robupy_obj = solve(get_robupy_obj(init_dict))

    # Distribute objects from solution. These are then used to reproduce the
    # EMAX calculations from inside the toolbox.
    periods_payoffs_systematic = robupy_obj.get_attr('periods_payoffs_systematic')
    states_number_period = robupy_obj.get_attr('states_number_period')
    mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')
    periods_emax = robupy_obj.get_attr('periods_emax')
    is_ambiguous = robupy_obj.get_attr('is_ambiguous')
    num_periods = robupy_obj.get_attr('num_periods')
    states_all = robupy_obj.get_attr('states_all')
    num_draws = robupy_obj.get_attr('num_draws')
    edu_start = robupy_obj.get_attr('edu_start')
    edu_max = robupy_obj.get_attr('edu_max')
    version = robupy_obj.get_attr('version')
    shocks = robupy_obj.get_attr('shocks')
    delta = robupy_obj.get_attr('delta')
    level = robupy_obj.get_attr('level')

    # Derived attributes
    current_period = num_periods - 3
    current_index = int(states_number_period[current_period]*0.5)
    future_period = current_period + 1

    # Extract the agent characteristics from baseline state and then collect
    # the indicators for all admissible future states. I choose the following
    # order throughout the project: Occupation A, Occupation B, School, Home.
    exp_A, exp_B, edu, edu_lagged = states_all[current_period, current_index, :]

    future_indices = []
    future_indices += [mapping_state_idx[future_period, exp_A + 1, exp_B, edu,
                                           0]]
    future_indices += [mapping_state_idx[future_period, exp_A, exp_B + 1, edu,
                                           0]]
    future_indices += [mapping_state_idx[future_period, exp_A, exp_B, edu + 1,
                                           1]]
    future_indices += [mapping_state_idx[future_period, exp_A, exp_B, edu,
                                           0]]

    # Construct current ex post payoffs. I choose these manually to generate
    # a relevant, interesting choice problem.
    payoff_systematic_current = \
        periods_payoffs_systematic[current_period, current_index, :]

    payoffs_ex_post_current = []
    for j in [0, 1]:
        payoffs_ex_post_current += [payoff_systematic_current[j] * np.exp(0)]
    for j in [2, 3]:
        payoffs_ex_post_current += [payoff_systematic_current[j] + 0]

    # Iterate over the four alternative decisions for an agent at the
    # beginning of a decision tree.
    for future_index, state_indicator in enumerate(future_indices):
        # Extract systematic components for the future components as these
        # are relevant for the EMAX calculation.
        payoffs_systematic_future = \
            periods_payoffs_systematic[future_period, state_indicator, :]
        # Determine the maximum and minimum value of the value function.
        bounds = []
        for sign in [1, -1]:
            # Select disturbances for next period, which are relevant for
            # the EMAX calculation.
            periods_eps_relevant = create_disturbances(robupy_obj, False)
            eps_relevant = periods_eps_relevant[future_period, :, :]
            # In the risk-only case, no modification of the distribution of
            # the disturbances is required.
            eps_relevant_emax = eps_relevant
            # In the case of ambiguity, the upper and lower bounds for the
            # value function are required.
            if is_ambiguous:
                # Set up the arguments for the optimization request.
                x0, options = [0.0, 0.0], dict()
                options['maxiter'] = 100000000
                constraints = _prep_kl(shocks, level)
                args = (num_draws, eps_relevant, future_period, state_indicator,
                        payoffs_systematic_future, edu_max, edu_start,
                        mapping_state_idx, states_all, num_periods,
                        periods_emax, delta, sign)
                # Run the optimization to determine upper and lower bound
                # depending on the sign variable.
                opt = minimize(criterion, x0, args, method='SLSQP',
                        options=options, constraints=constraints)
                # Check success
                assert (opt['success'] == True)
                # Transform relevant disturbances according to results from
                # optimization step.
                eps_relevant_emax = \
                    transform_disturbances_ambiguity(eps_relevant, opt['x'])

            # Back out the relevant EMAX using the results from the
            # optimization.
            emax_simulated, _, _ = simulate_emax(num_periods, num_draws,
                future_period, state_indicator, eps_relevant_emax,
                payoffs_systematic_future, edu_max, edu_start, periods_emax,
                states_all, mapping_state_idx, delta)

            # Construct total payoffs of admissible future states.
            total_payoffs = payoffs_ex_post_current[future_index] + \
                                delta * emax_simulated

            # Collect results in tuples for each level of ambiguity.
            bounds += [total_payoffs]

            # This is a manual test for the EMAX calculations. These should
            # be identical for the standard worst-case evaluation. Only
            # working for the PYTHON implementation as the disturbances need
            # ot be aligned.
            if sign == 1 and version == 'PYTHON':
                np.testing.assert_almost_equal(emax_simulated,
                    periods_emax[future_period, state_indicator])

        # Collect bounds
        rslt += [bounds]

    # Cleanup
    os.chdir('../'), os.chdir('../')

    # Finishing
    return rslt

''' Execution of module as script.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Assess implications of model misspecification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--levels', action='store', type=float, dest='levels',
        required=True, nargs='+', help='level of ambiguity in true economy')

    parser.add_argument('--recompile', action='store_true', default=False,
        dest='is_recompile', help='recompile package')

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        help='only three periods')

    parser.add_argument('--procs', action='store', type=int, dest='num_procs',
         default=1, help='use multiple processors')

    # Cleanup
    os.system('./clean'), os.mkdir('rslts')

    # Distribute attributes
    levels, is_recompile, is_debug, num_procs = distribute_arguments(parser)

    # Read the baseline specification and obtain the initialization dictionary.
    shutil.copy(SPEC_DIR + '/data_one.robupy.ini', 'model.robupy.ini')
    init_dict = read('model.robupy.ini').get_attr('init_dict')
    os.unlink('model.robupy.ini')

    # Ensure that fast version of package is available. This is a little more
    # complicated than usual as the compiler on acropolis does use other
    # debugging flags and thus no debugging is requested.
    if is_recompile:
        if 'acropolis' in socket.gethostname():
            compile_package('--fortran', True)
        else:
            compile_package('--fortran --debug', True)

    # Set up pool for processors for parallel execution.
    process_tasks = partial(run, init_dict, is_debug)
    bounds = Pool(num_procs).map(process_tasks, levels)

    # Restructure return arguments for better interpretability and further
    # processing.
    rslt = dict()
    for i, level in enumerate(levels):
        rslt[level] = bounds[i]

    # Store for further processing
    pkl.dump(rslt, open('rslts/admissible_values.robupy.pkl', 'wb'))



