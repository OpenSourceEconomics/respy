""" This module contains functionality that is shared between the solution
and simulation modules.
"""

# standard library
import numpy as np


def replace_missing_values(argument):
    """ Replace missing value -99 with NAN. Note that the output argument is
    of type float.
    """
    # Determine missing values
    is_missing = (argument == -99)

    # Transform to float array
    argument = np.asfarray(argument)

    # Replace missing values
    argument[is_missing] = np.nan

    # Finishing
    return argument


def read_disturbances(robupy_obj):
    """ Red the disturbances from disk. This is only used in the development
    process.
    """
    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')

    num_draws = robupy_obj.get_attr('num_draws')

    # Initialize containers
    periods_eps_relevant = np.tile(np.nan, (num_periods, num_draws, 4))

    # Read and distribute disturbances
    disturbances = np.array(np.genfromtxt('disturbances.txt'), ndmin=2)
    for period in range(num_periods):
        lower = 0 + num_draws*period
        upper = lower + num_draws
        periods_eps_relevant[period, :, :] = disturbances[lower:upper, :]

    # Finishing
    return periods_eps_relevant
