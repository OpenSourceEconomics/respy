""" This module contains some auxiliary functions for the simulation from the
model.
"""

# standard library
import numpy as np
import logging

import pandas as pd

from robupy.shared.constants import MISSING_FLOAT
from robupy.shared.auxiliary import get_total_value

logger = logging.getLogger('ROBUPY_SIMULATE')

def simulate_sample(num_agents, states_all, num_periods, mapping_state_idx,
        periods_payoffs_systematic, periods_draws_emax, edu_max, edu_start,
        periods_emax, delta):
    """ Sample simulation
    """
    count = 0

    # Initialize data
    dataset = np.tile(MISSING_FLOAT, (num_agents * num_periods, 8))

    for i in range(num_agents):

        current_state = states_all[0, 0, :].copy()

        dataset[count, 0] = i

        # Logging
        if (i != 0) and (i % 100 == 0):
            logger.info('... simulated ' + str(i) + ' agents')

        # Iterate over each period for the agent
        for period in range(num_periods):

            # Distribute state space
            exp_a, exp_b, edu, edu_lagged = current_state

            k = mapping_state_idx[period, exp_a, exp_b, edu, edu_lagged]

            # Write agent identifier and current period to data frame
            dataset[count, :2] = i, period

            # Select relevant subset
            payoffs_systematic = periods_payoffs_systematic[period, k, :]
            draws = periods_draws_emax[period, i, :]

            # Get total value of admissible states
            total_payoffs, payoffs_ex_post, _ = get_total_value(period,
                num_periods, delta, payoffs_systematic, draws, edu_max,
                edu_start, mapping_state_idx, periods_emax, k, states_all)

            # Determine optimal choice
            max_idx = np.argmax(total_payoffs)

            # Record agent decision
            dataset[count, 2] = max_idx + 1

            # Record earnings
            dataset[count, 3] = MISSING_FLOAT
            if max_idx in [0, 1]:
                dataset[count, 3] = payoffs_ex_post[max_idx]

            # Write relevant state space for period to data frame
            dataset[count, 4:8] = current_state

            # Special treatment for education
            dataset[count, 6] += edu_start

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

    # Finishing
    return dataset


def write_info(robupy_obj, data_frame):
    """ Write information about the simulated economy.
    """
    # Distribute class attributes
    file_sim = robupy_obj.get_attr('file_sim')

    seed = robupy_obj.get_attr('seed_data')

    # Get basic information
    num_agents = data_frame[1].value_counts()[0]

    num_periods = data_frame[0].value_counts()[0]

    # Write information to file
    with open(file_sim + '.info', 'w') as file_:

        file_.write('\n Simulated Economy\n\n')

        file_.write('   Number of Agents:       ' + str(num_agents) + '\n\n')

        file_.write('   Number of Periods:      ' + str(num_periods) + '\n\n')

        file_.write('   Seed:                   ' + str(seed) + '\n\n\n')

        file_.write('   Choices\n\n')

        file_.write('       Period     Work A     Work B    '
                    'Schooling   Home      \n\n')

        for t in range(num_periods):

            work_a = np.sum((data_frame[2] == 1) &
                            (data_frame[1] == t))/num_agents

            work_b = np.sum((data_frame[2] == 2) & (data_frame[1] ==
                                                    t))/num_agents

            schooling = np.sum((data_frame[2] == 3) &
                               (data_frame[1] == t))/num_agents

            home = np.sum((data_frame[2] == 4) & (data_frame[1] ==
                                                  t))/num_agents

            string = '''{0[0]:>10}    {0[1]:10.4f} {0[2]:10.4f}
             {0[3]:10.4f} {0[4]:10.4f}\n'''

            args = [(t + 1), work_a, work_b, schooling, home]
            file_.write(string.format(args))

        file_.write('\n\n\n')

        # Additional information about the simulated economy
        string = '''       {0[0]:<25}    {0[1]:10.4f}\n'''

        file_.write('   Additional Information\n\n')

        stat = data_frame[data_frame.ix[:, 1] ==
                (num_periods - 1)].ix[:, 6].mean()
        file_.write(string.format(['Average Education', stat]))

        file_.write('\n')

        stat = data_frame[data_frame.ix[:, 1] ==
                (num_periods - 1)].ix[:, 4].mean()
        file_.write(string.format(['Average Experience A', stat]))

        stat = data_frame[data_frame.ix[:, 1] ==
                (num_periods - 1)].ix[:, 5].mean()
        file_.write(string.format(['Average Experience B', stat]))


def write_out(data_frame, file_sim):
    """ Write dataset to file.
    """
    formats = []

    formats += [_format_integer, _format_integer, _format_integer]

    formats += [_format_float, _format_integer, _format_integer]

    formats += [_format_integer, _format_integer]

    with open(file_sim + '.dat', 'w') as file_:

        data_frame.to_string(file_, index=False, header=None, na_rep='.',
                            formatters=formats)


def _format_float(x):
    """ Pretty formatting for floats
    """
    if pd.isnull(x):
        return '    .'
    else:
        return '{0:10.2f}'.format(x)


def _format_integer(x):
    """ Pretty formatting for integers.
    """
    if pd.isnull(x):
        return '    .'
    else:
        return '{0:<5}'.format(int(x))


def check_simulation(robupy_obj):
    """ Check integrity of simulation request.
    """
    # Checks
    assert (robupy_obj.get_attr('is_solved'))
    assert (robupy_obj.get_status())

    # Finishing
    return True