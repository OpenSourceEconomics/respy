""" Module that contains the auxiliary functions that help assessing the
quality of interpolation methods.
"""

# standard library
import pickle as pkl
import numpy as np

import shutil
import os

# project library
from robupy.python.py.auxiliary import get_total_value
from robupy.tests.random_init import print_random_dict
from robupy.auxiliary import create_disturbances

from robupy import solve
from robupy import read

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']
SPEC_DIR = ROBUPY_DIR + '/development/analyses/restud/specifications'


def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    num_procs = args.num_procs

    level = args.level

    # Check arguments
    assert (num_procs > 0)
    assert (isinstance(level, float))
    assert (level >= 0.00)

    # Finishing
    return level, num_procs


def process_tasks(task):
    """ Process tasks to simulate and solve.
    """
    # Distribute requested task
    spec, solution, level = task

    # Enter appropriate subdirectory
    os.chdir(solution), os.chdir('data_' + spec)

    # Copy relevant initialization file
    src = SPEC_DIR + '/data_' + spec + '.robupy.ini'
    tgt = 'model.robupy.ini'

    shutil.copy(src, tgt)

    # Prepare initialization file
    robupy_obj = read('model.robupy.ini')

    init_dict = robupy_obj.get_attr('init_dict')

    # Modification from baseline initialization file
    init_dict['AMBIGUITY']['level'] = level
    init_dict['SOLUTION']['store'] = True
    init_dict['PROGRAM']['debug'] = True

    # TODO: Remove
    init_dict['BASICS']['periods'] = 5

    # Decide for interpolation or full solution
    is_full = (solution == 'full')
    if is_full:
        init_dict['INTERPOLATION']['apply'] = False
    else:
        init_dict['INTERPOLATION']['apply'] = True

    # Finalize initialization file and solve model
    print_random_dict(init_dict)

    robupy_obj = read('test.robupy.ini')

    solve(robupy_obj)

    os.chdir('../../')


def aggregate_results(SPECIFICATIONS, SOLUTIONS):
    """ Aggregate results across the different specifications and solution
    methods.
    """
    # Initialize containers
    failure_counts = dict()

    # Iterate over specifications
    for spec in SPECIFICATIONS:

        # Initialize container
        results = dict()

        # Run full and approximate solutions
        for solution in SOLUTIONS:
            # Enter appropriate subdirectory and collect results.
            os.chdir(solution), os.chdir('data_' + spec)
            results[solution] = pkl.load(open('solution.robupy.pkl', 'rb'))
            # Return to baseline
            os.chdir('../../')

        # Distribute class attributes
        full_rslts, approximate_rslts = results['full'], results['approximate']

        # Create disturbances for simulation step.
        periods_eps_relevant = create_disturbances(full_rslts, False)

        # Distribute class information
        periods_payoffs_systematic = full_rslts.get_attr('periods_payoffs_systematic')
        mapping_state_idx = full_rslts.get_attr('mapping_state_idx')
        num_periods = full_rslts.get_attr('num_periods')
        num_agents = full_rslts.get_attr('num_agents')
        states_all = full_rslts.get_attr('states_all')
        edu_start = full_rslts.get_attr('edu_start')
        edu_max = full_rslts.get_attr('edu_max')
        delta = full_rslts.get_attr('delta')

        # Extract alternative results for the expected future values
        emax_approximate = approximate_rslts.get_attr('periods_emax')
        emax_full = full_rslts.get_attr('periods_emax')

        # Construct failure counts
        failure_counts[spec] = _assess_approximation(num_agents, states_all,
            num_periods, mapping_state_idx, periods_payoffs_systematic,
            periods_eps_relevant, edu_max, edu_start, delta, emax_full,
            emax_approximate)

    # Store results for further processing
    pkl.dump(failure_counts, open('rslts.quality.pkl', 'wb'))


def _assess_approximation(num_agents, states_all, num_periods,
        mapping_state_idx, periods_payoffs_systematic, periods_eps_relevant,
        edu_max, edu_start, delta, periods_emax_full, periods_emax_approximate):
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
                mapping_state_idx, periods_emax_full, k, states_all)

            # Get total value of admissible states using interpolated solution
            total_payoffs_interpolated, _, _ = get_total_value(period,
                num_periods, delta, payoffs_systematic, disturbances, edu_max,
                edu_start, mapping_state_idx, periods_emax_approximate, k,
                states_all)

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