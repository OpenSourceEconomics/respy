""" This module contains auxiliary functions for the PYTEST suite.
"""

import numpy as np
import shlex

from respy.python.solve.solve_auxiliary import pyth_create_state_space
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy import RespyCls


def compare_est_log(base_est_log):
    """ This function is required as the log files can be slightly different
    for good reasons. The error capturing of an IndexError is required as sometimes the
    """

    for i in range(25):

        try:

            with open('est.respy.log') as in_file:
                alt_est_log = in_file.readlines()

            for i, _ in enumerate(alt_est_log):
                alt_line, base_line = alt_est_log[i], base_est_log[i]
                list_ = shlex.split(alt_line)

                if not list_:
                    continue

                if list_[0] in ['Criterion']:
                    alt_val = float(shlex.split(alt_line)[1])
                    base_val = float(shlex.split(base_line)[1])
                    np.testing.assert_almost_equal(alt_val, base_val)
                elif list_[0] in ['Time']:
                    pass
                else:
                    assert alt_line == base_line

            return

        except IndexError:
            pass

def write_interpolation_grid(file_name):
    """ Write out an interpolation grid that can be used across
    implementations.
    """
    # Process relevant initialization file
    respy_obj = RespyCls(file_name)

    # Distribute class attribute
    num_periods, num_points_interp, edu_start, edu_max, min_idx = \
        dist_class_attributes(respy_obj,
            'num_periods', 'num_points_interp', 'edu_start', 'edu_max', 'min_idx')

    # Determine maximum number of states
    _, states_number_period, _, max_states_period = \
        pyth_create_state_space(num_periods, edu_start, edu_max, min_idx)

    # Initialize container
    booleans = np.tile(True, (max_states_period, num_periods))

    # Iterate over all periods
    for period in range(num_periods):

        # Construct auxiliary objects
        num_states = states_number_period[period]
        any_interpolation = (num_states - num_points_interp) > 0

        # Check applicability
        if not any_interpolation:
            continue

        # Draw points for interpolation
        indicators = np.random.choice(range(num_states),
            size=(num_states - num_points_interp), replace=False)

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
