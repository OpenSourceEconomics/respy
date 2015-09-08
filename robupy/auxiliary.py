""" This module contains functionality that is shared between the solution
and simulation modules.
"""

# standard library
import numpy as np


def replace_missing_values(argument):
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


def read_restud_disturbances(robupy_obj):
    """ Red the disturbances from the RESTUD program. This is only used in
    the development process.
    """
    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')

    num_draws = robupy_obj.get_attr('num_draws')

    # Initialize containers
    periods_eps_relevant = np.tile(np.nan, (num_periods, num_draws, 4))

    # Read and distribute disturbances
    disturbances = np.array(np.genfromtxt('disturbances.txt'), ndmin = 2)
    for period in range(num_periods):
        lower = 0 + num_draws*period
        upper = lower + num_draws
        periods_eps_relevant[period, :, :] = disturbances[lower:upper, :]

    # Finishing
    return periods_eps_relevant