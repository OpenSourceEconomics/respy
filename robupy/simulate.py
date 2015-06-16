""" This module contains all the capabilities to simulate from a model.
"""

# standard library
import numpy as np
import pandas as pd

''' Public function
'''


def simulate(init_dict, k_state, emax, payoffs_ex_ante, f_state):
    """ Simulate from dynamic programming model.
    """

    num_periods = init_dict['BASICS']['periods']
    num_agents = init_dict['BASICS']['agents']

    # Initialization of COMPUTATION
    edu_start = init_dict['EDUCATION']['initial']

    # Disturbance
    # Draw random variable
    np.random.seed(init_dict['COMPUTATION']['seed'])

    eps = np.random.multivariate_normal(np.zeros(4), init_dict['SHOCKS'],
                                        (num_periods, num_agents))


    data = np.tile(np.nan, (num_agents * num_periods, 6))

    count = 0

    for i in range(num_agents):

        current_state = k_state[0, 0, :]

        data[count, 0] = i

        for period in range(num_periods):

            data[count, :2] = i, period

            # Select disturbances
            eps_agent = eps[period, i]

            # Get payoffs
            v = get_payoffs(period, current_state, emax, payoffs_ex_ante,
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

            data[count, 2:6] = current_state

            count += 1


    # Write to file
    formats = []

    formats += [_format_integer]

    formats += [_format_integer]

    formats += [_format_integer, _format_integer, _format_integer, _format_integer]

    df = pd.DataFrame(data)

    with open('data.robust.dat', 'w', newline='') as file_:

        df.to_string(file_, index=False, header=None, na_rep='.',
                     formatters=formats)


''' Private functions
'''


def get_payoffs(period, k_state, emax, payoffs_ex_ante, eps, f_state, edu_start):
        """ Get the alternative specific payoffs.

        """
        # Distribute
        exp_A, exp_B = k_state[0], k_state[1]
        edu, edu_lagged = k_state[2], k_state[3]

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
