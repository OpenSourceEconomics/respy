""" This module contains auxiliary functions for the PYTEST suite.
"""

# standard library
import numpy as np

# project library
from respy.python.solve.solve_auxiliary import pyth_create_state_space
from respy.python.shared.shared_auxiliary import dist_class_attributes

from respy import read

''' Auxiliary functions.
'''


def write_interpolation_grid(file_name):
    """ Write out an interpolation grid that can be used across
    implementations.
    """
    # Process relevant initialization file
    respy_obj = read(file_name)

    # Distribute class attribute
    num_periods, num_points, edu_start, edu_max, min_idx = \
        dist_class_attributes(respy_obj,
            'num_periods', 'num_points', 'edu_start', 'edu_max', 'min_idx')

    # Determine maximum number of states
    _, states_number_period, _, max_states_period = \
        pyth_create_state_space(num_periods, edu_start, edu_max, min_idx)

    # Initialize container
    booleans = np.tile(True, (max_states_period, num_periods))

    # Iterate over all periods
    for period in range(num_periods):

        # Construct auxiliary objects
        num_states = states_number_period[period]
        any_interpolation = (num_states - num_points) > 0

        # Check applicability
        if not any_interpolation:
            continue

        # Draw points for interpolation
        indicators = np.random.choice(range(num_states),
            size=(num_states - num_points), replace=False)

        # Replace indicators
        for i in range(num_states):
            if i in indicators:
                booleans[i, period] = False

    # Write out to file
    np.savetxt('interpolation.txt', booleans, fmt='%s')

    # Some information that is useful elsewhere.
    return max_states_period


def write_draws(num_periods, max_draws):
    """ Write out draws to potentially align the different implementations of
    the model. Note that num draws has to be less or equal to the largest
    number of requested random deviates.
    """
    # Draw standard deviates
    draws_standard = np.random.multivariate_normal(np.zeros(4),
        np.identity(4), (num_periods, max_draws))

    # Write to file to they can be read in by the different implementations.
    with open('draws.txt', 'w') as file_:
        for period in range(num_periods):
            for i in range(max_draws):
                fmt = ' {0:15.10f} {1:15.10f} {2:15.10f} {3:15.10f}\n'
                line = fmt.format(*draws_standard[period, i, :])
                file_.write(line)

