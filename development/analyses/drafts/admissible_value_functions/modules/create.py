#!/usr/bin/env python
""" This module creates a graphical illustration of all admissible values.
The idea is to use this to split the discussion between all admissible values
and really making a decision.
"""

# standard library
from scipy.optimize import minimize
from multiprocessing import Pool
from functools import partial


import pickle as pkl

import argparse
import shutil
import socket
import sys
import os

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']
SPEC_DIR = ROBUPY_DIR + '/development/analyses/restud/specifications'

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR + '/development/tests/random')
sys.path.insert(0, ROBUPY_DIR)

import numpy as np

# project library
from robupy import solve
from robupy import read

from robupy.python.py.ambiguity import transform_disturbances_ambiguity
from robupy.python.py.auxiliary import simulate_emax
from robupy.python.py.ambiguity import _prep_kl

from robupy.auxiliary import create_disturbances

from auxiliary import distribute_arguments
from auxiliary import get_robupy_obj
from auxiliary import criterion

from modules.auxiliary import compile_package


################################################################################
################################################################################
def run(init_dict, is_debug, ambiguity_level):

    # Auxiliary objects
    name = '%03.3f' % ambiguity_level
    period, rslt = 1, []

    # Prepare directory structure
    os.mkdir(name), os.chdir(name)

    # Initialize auxiliary objects
    init_dict['AMBIGUITY']['level'] = ambiguity_level

    # Restrict number of periods for debugging purposes.
    if is_debug:
        init_dict['BASICS']['periods'] = 3

    # Solve the basic economy
    robupy_obj = solve(get_robupy_obj(init_dict))

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
    version = robupy_obj.get_attr('version')
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
            # the disturbances is required. This case is only handled in
            # this script for benchmarking purposes.
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
                opt = minimize(criterion, x0, args, method='SLSQP', options=options,
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
            if sign == 1 or ambiguity_level == 0.0:
                if version == 'PYTHON':
                    np.testing.assert_almost_equal(emax_simulated,
                        periods_emax[period, state_indicator])

        # Collect bounds
        rslt += [bounds]

    # Cleanup
    os.chdir('../')

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
    os.system('./clean')

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
    pkl.dump(rslt, open('admissible.robupy.pkl', 'wb'))



