""" This module contains all the capabilities to simulate from the model.

    Structure of Dataset:

        0   Identifier of Agent
        1   Time Period
        2   Choice (1 = Work A, 2 = Work B, 3 = Education, 4 = Home)
        3   Earnings (missing value if not working)
        4   Work Experience A
        5   Work Experience B
        6   Schooling
        7   Lagged Schooling

"""

# standard library
import pandas as pd
import numpy as np
import logging
import os

# project library
from robupy.checks.checks_simulate import checks_simulate
from robupy.auxiliary import replace_missing_values
from robupy.auxiliary import read_restud_disturbances

import robupy.python.py.python_core as python_core
try:
    import robupy.python.f2py.f2py_core as f2py_core
except ImportError:
    pass

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
    num_agents = robupy_obj.get_attr('seed_simulation')

    seed = robupy_obj.get_attr('num_agents')

    # Draw disturbances for the simulation.
    periods_eps_relevant = _create_eps(robupy_obj)

    # Simulate a dataset with the results from the solution and write out the
    # dataset to a text file. In addition a file summarizing the dataset is
    # produced.
    logger.info('Staring simulation of model for ' + str(num_agents) +
        ' agents with seed ' + str(seed))

    data_frame = _wrapper_simulate_sample(robupy_obj, periods_eps_relevant)

    _write_out(data_frame)

    _write_info(robupy_obj, data_frame)

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

    debug = robupy_obj.get_attr('debug')

    fast = robupy_obj.get_attr('version')

    # Auxiliary object
    is_f2py = (fast == 'F2PY')

    # Interface to core functions
    if fast:
        data_frame = f2py_core.wrapper_simulate_sample(num_agents,
            states_all, num_periods, mapping_state_idx, periods_payoffs_ex_ante,
            periods_eps_relevant, edu_max, edu_start, periods_emax, delta)
    else:
        data_frame = python_core.simulate_sample(num_agents, states_all,
            num_periods, mapping_state_idx, periods_payoffs_ex_ante,
            periods_eps_relevant, edu_max, edu_start, periods_emax, delta)

    # Replace missing values
    data_frame = replace_missing_values(data_frame)

    # Create pandas data frame
    data_frame = pd.DataFrame(data_frame)

    # Run checks on data frame
    if debug:
        checks_simulate('data_frame', robupy_obj, data_frame)

    # Finishing
    return data_frame

''' Auxiliary functions
'''


def _create_eps(robupy_obj):
    """ Create relevant disturbances.
    """
    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')

    num_agents = robupy_obj.get_attr('num_agents')

    seed = robupy_obj.get_attr('seed_simulation')

    shocks = robupy_obj.get_attr('shocks')

    debug = robupy_obj.get_attr('debug')

    # Set random seed
    np.random.seed(seed)

    # Draw random disturbances and adjust them for the two occupations
    np.random.seed(seed)
    periods_eps_relevant = np.random.multivariate_normal(np.zeros(4),
        shocks, (num_periods, num_agents))
    for period in range(num_periods):
        for j in [0, 1]:
            periods_eps_relevant[period, :, j] = np.exp(periods_eps_relevant[
                                                  period, :, j])

    # This is only used to compare the RESTUD program to the ROBUPY package.
    # It aligns the random components between the two. It is only used in the
    # development process.
    if debug and os.path.isfile('disturbances.txt'):
        periods_eps_relevant = read_restud_disturbances(robupy_obj)

    # Finishing
    return periods_eps_relevant


def _write_info(robupy_obj, data_frame):
    """ Write information about the simulated economy.
    """
    # Distribute class attributes
    seed = robupy_obj.get_attr('seed_simulation')

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
    # Formatting of columns
    formats = []

    formats += [_format_integer, _format_integer, _format_integer]

    formats += [_format_float, _format_integer, _format_integer]

    formats += [_format_integer, _format_integer]

    with open('data.robupy.dat', 'w') as file_:

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
