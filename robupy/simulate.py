""" This module contains all the capabilities to simulate from the model.

    Structure of Dataset:

    0   Identifier of Agent
    1   Time Period
    2   Choice (0 = Work A, 1 = Work B, 2 = Education, 3 = Home)
    3   Earnings (missing value if not working)
    4   Work Experience A
    5   Work Experience B
    6   Schooling
    7   Lagged Schooling

"""

# standard library
import numpy as np
import pandas as pd
import logging

# project library
import robupy.performance.access as perf

from robupy.checks.checks_simulate import checks_simulate

# Logging
logger = logging.getLogger('ROBUPY_SIMULATE')

''' Public function
'''


def simulate(robupy_obj):
    """ Simulate from dynamic programming model.
    """
    # Antibugging
    assert (robupy_obj.get_status())

    # Distribute class attributes
    debug = robupy_obj.get_attr('debug')

    shocks = robupy_obj.get_attr('shocks')

    states_all = robupy_obj.get_attr('states_all')

    emax = robupy_obj.get_attr('emax')

    delta = robupy_obj.get_attr('delta')

    edu_start = robupy_obj.get_attr('edu_start')

    edu_max = robupy_obj.get_attr('edu_max')

    period_payoffs_ex_ante = robupy_obj.get_attr('period_payoffs_ex_ante')

    mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')

    num_periods = robupy_obj.get_attr('num_periods')

    num_agents = robupy_obj.get_attr('num_agents')

    seed = robupy_obj.get_attr('seed_simulation')

    fast = robupy_obj.get_attr('fast')

    # Access performance library
    perf_lib = perf.get_library(fast)

    # Logging
    logger.info('Staring simulation of model for ' +
                str(num_agents) + ' agents with seed ' + str(seed))

    # Draw disturbances.
    np.random.seed(seed)

    eps = np.random.multivariate_normal(np.zeros(4), shocks,
                                        (num_periods, num_agents))

    # Initialize data
    data = np.tile(np.nan, (num_agents * num_periods, 8))

    # Initialize row indicator
    count = 0

    for i in range(num_agents):

        current_state = states_all[0, 0, :].copy()

        data[count, 0] = i

        # Logging
        if (i != 0) and (i % 100 == 0):
            logger.info('... simulated ' + str(i) + ' agents')

        # Iterate over each period for the agent
        for period in range(num_periods):

            # Distribute state space
            exp_A, exp_B, edu, edu_lagged = current_state

            k = mapping_state_idx[period, exp_A, exp_B, edu, edu_lagged]

            # Write agent identifier and current period to data frame
            data[count, :2] = i, period

            period_payoffs_ex_post = np.tile(np.nan, 4)

            # Calculate ex post payoffs
            for j in [0, 1]:
                period_payoffs_ex_post[j] = period_payoffs_ex_ante[period, k, j] * \
                                                np.exp(eps[period, i, j])

            for j in [2, 3]:
                period_payoffs_ex_post[j] = period_payoffs_ex_ante[period, k, j] + \
                                                eps[period, i, j]

            # Calculate future utilities
            if period == (num_periods - 1):
                future_payoffs = np.zeros(4)
            else:
                future_payoffs = perf_lib.get_future_payoffs(edu_max, edu_start,
                                                             mapping_state_idx,
                                                             period, emax, k,
                                                             states_all)

            # Calculate total utilities
            total_payoffs = period_payoffs_ex_post + delta * future_payoffs

            # Determine optimal choice
            max_idx = np.argmax(total_payoffs)

            # Record agent decision
            data[count, 2] = max_idx

            # Record earnings
            data[count, 3] = np.nan
            if max_idx in [0, 1]:
                data[count, 3] = period_payoffs_ex_post[max_idx]

            # Write relevant state space for period to data frame
            data[count, 4:8] = current_state

            # Special treatment for education
            data[count, 6] += edu_start

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

            # Update row indicator
            count += 1

    # Convert to pandas data frame
    data_frame = pd.DataFrame(data)

    # Run checks of pandas data array
    if debug is True:
        checks_simulate('data_frame', robupy_obj, data_frame)

    # Write to dataset and information to file
    _write_out(data_frame)

    _write_info(data_frame, seed)

    # Logging
    logger.info('... finished \n')

    # Finishing
    return data_frame

''' Private functions
'''

def _write_info(data_frame, seed):
    """ Write information about the simulated economy.
    """

    # Get basic information
    num_agents = data_frame[1].value_counts()[0]

    num_periods = data_frame[0].value_counts()[0]

    # Write information to file
    with open('data.robupy.info', 'w') as file_:

        file_.write('\n Simulated Economy\n\n')

        file_.write('   Number of Agents:       ' + str(num_agents) + '\n\n')

        file_.write('   Number of Periods:      ' + str(num_periods) + '\n\n')

        file_.write('   Seed:                   ' + str(seed) + '\n\n\n')

        file_.write('   Choices:  \n\n')

        file_.write('       Period     Work A     Work B    Schooling   Home      \n\n')

        for t in range(num_periods):

            work_a = np.sum((data_frame[2] == 0) & (data_frame[1] ==
                                                        t))/num_agents

            work_b = np.sum((data_frame[2] == 1) & (data_frame[1] == t))/num_agents

            schooling = np.sum((data_frame[2] == 2) & (data_frame[1] ==
                                                           t))/num_agents

            home = np.sum((data_frame[2] == 3) & (data_frame[1] == t))/num_agents

            string = '''{0[0]:>10}    {0[1]:10.4f} {0[2]:10.4f} {0[3]:10.4f} {0[4]:10.4f}\n'''

            file_.write(string.format([(t + 1), work_a, work_b, schooling, home]))

        file_.write('\n\n')


def _write_out(data_frame):
    """ Write dataset to file.
    """
    # Antibugging
    assert (isinstance(data_frame, pd.DataFrame))

    # Formatting of columns
    formats = []

    formats += [_format_integer, _format_integer, _format_integer]

    formats += [_format_float, _format_integer, _format_integer]

    formats += [_format_integer, _format_integer]

    with open('data.robupy.dat', 'w') as file_:

        data_frame.to_string(file_, index=False, header=None, na_rep='.',
                     formatters=formats)

def _format_float(x):
    """ Format floating point number.
    """
    if pd.isnull(x):

        return '    .'

    else:

        return '{0:10.2f}'.format(x)

def _format_integer(x):
    """ Format integers.
    """
    if pd.isnull(x):

        return '    .'

    else:

        return '{0:<5}'.format(int(x))
