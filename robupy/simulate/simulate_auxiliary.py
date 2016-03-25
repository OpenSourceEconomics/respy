""" This module contains some auxiliary functions for the simulation from the
model.
"""

# standard library
import numpy as np
import logging

import pandas as pd
logger = logging.getLogger('ROBUPY_SIMULATE')

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

            string = '''{0[0]:>10}    {0[1]:10.4f} {0[2]:10.4f} {0[3]:10.4f} {0[4]:10.4f}\n'''

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

    # Write out the parametrization of the simulated economy.
    model_paras = robupy_obj.get_attr('model_paras')
    vector = get_estimation_vector(model_paras)
    np.savetxt(open('data.robupy.paras', 'wb'), vector, fmt='%15.8f')


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

def get_estimation_vector(model_paras):
    """ Construct the vector estimation arguments.
    """

    # Distribute auxiliary objects
    shocks_cholesky = model_paras['shocks_cholesky']

    # Collect parameters
    vector = list()

    vector += model_paras['coeffs_a'].tolist()

    vector += model_paras['coeffs_b'].tolist()

    vector += model_paras['coeffs_edu'].tolist()

    vector += model_paras['coeffs_home'].tolist()

    vector += shocks_cholesky[0:4, 0].tolist()

    vector += shocks_cholesky[1:4, 1].tolist()

    vector += shocks_cholesky[2:4, 2].tolist()

    vector += shocks_cholesky[3:4, 3].tolist()

    # Finishing
    return vector


def check_simulation(robupy_obj):
    """ Check integrity of simulation request.
    """
    # Checks
    assert robupy_obj.get_attr('is_solved')
    assert robupy_obj.get_attr('is_locked')

    # Finishing
    return True