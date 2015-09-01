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
from robupy.checks.checks_simulate import checks_simulate
from robupy.performance.python.auxiliary import get_future_payoffs

import robupy.performance.fortran.fortran_core as fortran_core
import robupy.performance.python.python_core as python_core


# Logging
logger = logging.getLogger('ROBUPY_SIMULATE')

def _replace_missing_values(argument):
    """ Replace missing value -99 with NAN
    """
    # Determine missing values
    is_missing = (argument == -99)

    # Transform to float array
    mapping_state_idx = np.asfarray(argument)

    # Replace missing values
    mapping_state_idx[is_missing] = np.nan

    # Finishing
    return mapping_state_idx


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

    num_periods = robupy_obj.get_attr('num_periods')

    num_agents = robupy_obj.get_attr('num_agents')

    seed = robupy_obj.get_attr('seed_simulation')

    # Draw disturbances for the simulation.
    np.random.seed(seed)

    periods_eps_relevant = np.random.multivariate_normal(np.zeros(4), shocks,
                                        (num_periods, num_agents))

    # Simulate a dataset with the results from the solution.
    logger.info('Staring simulation of model for ' +
                str(num_agents) + ' agents with seed ' + str(seed))

    data_frame = _wrapper_simulate_sample(robupy_obj, periods_eps_relevant)

    # Run checks of pandas data array
    if debug:
        checks_simulate('data_frame', robupy_obj, data_frame)

    # Write to dataset and information to file
    _write_out(data_frame)

    _write_info(data_frame, seed)

    # Logging
    logger.info('... finished \n')

    # Finishing
    return data_frame

''' Wrappers for core functions
'''


def _wrapper_simulate_sample(robupy_obj, periods_eps_relevant):
    """ Wrapper for PYTHON and FORTRAN implementation of sample simulation.
    """
    # Distribute class attributes
    periods_payoffs_ex_ante = robupy_obj.get_attr('periods_payoffs_ex_ante')

    mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')

    periods_emax = robupy_obj.get_attr('periods_emax')

    num_periods = robupy_obj.get_attr('num_periods')

    states_all = robupy_obj.get_attr('states_all')

    num_agents = robupy_obj.get_attr('num_agents')

    edu_start = robupy_obj.get_attr('edu_start')

    edu_max = robupy_obj.get_attr('edu_max')

    delta = robupy_obj.get_attr('delta')

    fast = robupy_obj.get_attr('fast')

    # Interface to core functions
    if fast:
        dataset = fortran_core.simulate_sample(num_agents, states_all,
            num_periods, mapping_state_idx, periods_payoffs_ex_ante,
            periods_eps_relevant, edu_max, edu_start, periods_emax, delta)
    else:
        dataset = python_core.simulate_sample(num_agents, states_all,
            num_periods, mapping_state_idx, periods_payoffs_ex_ante,
            periods_eps_relevant, edu_max, edu_start, periods_emax, delta)

    # Set missing values to NAN
    dataset = _replace_missing_values(dataset)

    # Create pandas data frame
    dataset = pd.DataFrame(dataset)

    # Finishing
    return dataset

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

            work_a = np.sum((data_frame[2] == 1) & (data_frame[1] ==
                                                        t))/num_agents

            work_b = np.sum((data_frame[2] == 2) & (data_frame[1] ==
                                                    t))/num_agents

            schooling = np.sum((data_frame[2] == 3) & (data_frame[1] ==
                                                           t))/num_agents

            home = np.sum((data_frame[2] == 4) & (data_frame[1] ==
                                                  t))/num_agents

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
