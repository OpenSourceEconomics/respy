""" This module contains all the capabilities to simulate from the model.
"""

# standard library
import numpy as np
import pandas as pd

''' Public function
'''


def simulate(robupy_obj):
    """ Simulate from dynamic programming model.
    """
    # Antibugging
    assert (robupy_obj.get_status())

    # Distribute class attributes
    shocks = robupy_obj.get_attr('shocks')

    k_state = robupy_obj.get_attr('k_state')

    emax = robupy_obj.get_attr('emax')

    payoffs_ex_ante = robupy_obj.get_attr('payoffs_ex_ante')

    f_state = robupy_obj.get_attr('f_state')

    num_periods = robupy_obj.get_attr('num_periods')

    num_agents = robupy_obj.get_attr('num_agents')

    edu_start = robupy_obj.get_attr('edu_start')

    seed = robupy_obj.get_attr('seed')

    # Draw disturbances
    np.random.seed(seed)

    eps = np.random.multivariate_normal(np.zeros(4), shocks,
                                        (num_periods, num_agents))

    # Initialize data
    data = np.tile(np.nan, (num_agents * num_periods, 6))

    # Initialize row indicator
    count = 0

    for i in range(num_agents):

        current_state = k_state[0, 0, :].copy()

        data[count, 0] = i

        # Iterate over each period for the agent
        for period in range(num_periods):

            # Write agent identifier and current period to data frame
            data[count, :2] = i, period

            # Select disturbances
            eps_agent = eps[period, i]

            # Get payoffs
            v = _get_payoffs(period, current_state, emax, payoffs_ex_ante,
                             eps_agent, f_state, edu_start)

            max_idx = np.argmax(v)

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

            # Write relevant state space for period to data frame
            data[count, 2:6] = current_state

            # Update row indicator
            count += 1

    # Write to file
    _write_out(data)


''' Private functions
'''


def _write_out(data):
    """ Write dataset to file.
    """
    # Antibugging
    assert (isinstance(data, np.ndarray))

    # Formatting of columns
    formats = []

    formats += [_format_integer, _format_integer, _format_integer]

    formats += [_format_integer, _format_integer, _format_integer]

    # Convert to pandas data frame
    df = pd.DataFrame(data)

    with open('data.robupy.dat', 'w', newline='') as file_:
        # noinspection PyPep8
        df.to_string(file_, index=False, header=None, na_rep='.',
                     formatters=formats)


def _get_payoffs(period, k_state, emax, payoffs_ex_ante, eps, f_state,
                 edu_start):
    """ Get the alternative specific payoffs.
    """
    # Distribute current state space
    exp_A, exp_B, edu, edu_lagged = k_state

    # Get index
    idx = f_state[period, exp_A, exp_B, (edu - edu_start), edu_lagged]

    # Construct total value
    values = payoffs_ex_ante[period, idx, :] + eps

    values += emax[period, idx]

    # Finishing
    return values


def _format_float(x):
    """ Format floating point number.
    """
    if pd.isnull(x):

        return '    .'

    else:

        return "%15.8f" % x


def _format_integer(x):
    """ Format integers.
    """
    if pd.isnull(x):

        return '    .'

    else:

        return "%5d" % x
