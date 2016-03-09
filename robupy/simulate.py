""" This module contains the interface to simulate a dataset and write the
information to disk.

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

# project library
import robupy.python.py.python_library as python_library

from robupy.auxiliary import replace_missing_values
from robupy.auxiliary import create_disturbances
from robupy.auxiliary import check_dataset

# Logging
logger = logging.getLogger('ROBUPY_SIMULATE')

''' Main function
'''


def simulate(robupy_obj):
    """ Simulate dataset from model.
    """
    # Checks
    assert _check_simulation(robupy_obj)

    # Distribute class attributes
    is_ambiguous = robupy_obj.get_attr('is_ambiguous')

    model_paras = robupy_obj.get_attr('model_paras')

    num_periods = robupy_obj.get_attr('num_periods')

    num_agents = robupy_obj.get_attr('num_agents')

    seed = robupy_obj.get_attr('seed_data')

    is_debug = robupy_obj.get_attr('is_debug')

    # Auxiliary objects
    eps_cholesky = model_paras['eps_cholesky']

    seed_data = robupy_obj.get_attr('seed_data')

    # Draw disturbances for the simulation.
    disturbances_data = create_disturbances(num_periods, num_agents, seed_data,
                                           is_debug, 'sims',
        eps_cholesky, is_ambiguous)

    # Simulate a dataset with the results from the solution and write out the
    # dataset to a text file. In addition a file summarizing the dataset is
    # produced.
    logger.info('Staring simulation of model for ' + str(num_agents) +
        ' agents with seed ' + str(seed))

    data_frame = _wrapper_simulate_sample(robupy_obj, disturbances_data)

    _write_out(data_frame)

    _write_info(robupy_obj, data_frame)

    logger.info('... finished \n')

    # Finishing
    return data_frame

''' Auxiliary functions
'''


def _wrapper_simulate_sample(robupy_obj, disturbances_data):
    """ Wrapper for PYTHON and F2PY implementation of sample simulation.
    """
    # Distribute class attributes
    periods_payoffs_systematic = robupy_obj.get_attr('periods_payoffs_systematic')

    mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')

    periods_emax = robupy_obj.get_attr('periods_emax')

    num_periods = robupy_obj.get_attr('num_periods')

    states_all = robupy_obj.get_attr('states_all')

    num_agents = robupy_obj.get_attr('num_agents')

    edu_start = robupy_obj.get_attr('edu_start')

    is_python = robupy_obj.get_attr('is_python')

    is_debug = robupy_obj.get_attr('is_debug')

    edu_max = robupy_obj.get_attr('edu_max')

    delta = robupy_obj.get_attr('delta')

    # Interface to core functions
    if is_python:
        data_frame = python_library.simulate_sample(num_agents, states_all,
                                                    num_periods, mapping_state_idx, periods_payoffs_systematic,
                                                    disturbances_data, edu_max, edu_start, periods_emax, delta)
    else:
        import robupy.python.f2py.f2py_library as f2py_library
        data_frame = f2py_library.wrapper_simulate_sample(num_agents,
                                                          states_all, num_periods, mapping_state_idx,
                                                          periods_payoffs_systematic, disturbances_data, edu_max,
                                                          edu_start, periods_emax, delta)

    # Replace missing values
    data_frame = replace_missing_values(data_frame)

    # Create pandas data frame
    data_frame = pd.DataFrame(data_frame)

    # Run checks on data frame
    if is_debug:
        check_dataset(data_frame, robupy_obj)

    # Finishing
    return data_frame


''' Auxiliary functions
'''


def _write_info(robupy_obj, data_frame):
    """ Write information about the simulated economy.
    """
    # Distribute class attributes
    seed = robupy_obj.get_attr('seed_data')

    # Get basic information
    num_agents = data_frame[1].value_counts()[0]

    num_periods = data_frame[0].value_counts()[0]

    # Write information to file
    with open('data.robupy.info', 'w') as file_:

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


def _write_out(data_frame):
    """ Write dataset to file.
    """
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


def _check_simulation(robupy_obj):
    """ Check integrity of simulation request.
    """
    # Checks
    assert (robupy_obj.get_attr('is_solved'))
    assert (robupy_obj.get_status())

    # Finishing
    return True