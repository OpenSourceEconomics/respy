""" This module contains functionality that is shared between the solution
and simulation modules.
"""

# standard library
import numpy as np
import os


def create_disturbances(robupy_obj, is_simulation):
    """ Create disturbances.  Handle special case of zero variances as this
    case is useful for hand-based testing. The disturbances are drawn from a
    standard normal distribution and transformed later in the code.
    """
    # Distribute class attributes
    eps_cholesky = robupy_obj.get_attr('eps_cholesky')

    is_ambiguous = robupy_obj.get_attr('is_ambiguous')

    num_periods = robupy_obj.get_attr('num_periods')

    if is_simulation:

        num_draws = robupy_obj.get_attr('num_agents')

        seed = robupy_obj.get_attr('seed_simulation')

    else:

        num_draws = robupy_obj.get_attr('num_draws')

        seed = robupy_obj.get_attr('seed_solution')

    debug = robupy_obj.get_attr('debug')

    # Initialize container
    periods_eps_relevant = np.tile(-99.00, (num_periods, num_draws, 4))

    # This allows to use the same random disturbances across the different
    # implementations of the mode, including the RESTUD program. Otherwise,
    # we draw a new set of standard deviations
    if debug and os.path.isfile('disturbances.txt'):
        standard_deviates = read_disturbances(robupy_obj)
        standard_deviates = standard_deviates[:num_periods, :num_draws, :]
    else:
        np.random.seed(seed)
        standard_deviates = np.random.multivariate_normal(np.zeros(4),
            np.identity(4), (num_periods, num_draws))

    # In the case of ambiguous world, the standard deviates are used in the
    # solution part of the program.
    if is_ambiguous and not is_simulation:
        periods_eps_relevant = standard_deviates
    else:
        # Transform disturbances to relevant distribution
        for period in range(num_periods):
            periods_eps_relevant[period, :, :] = np.dot(eps_cholesky,
                    standard_deviates[period, :, :].T).T
            for j in [0, 1]:
                periods_eps_relevant[period, :, j] = np.exp(periods_eps_relevant[
                                                          period, :, j])

    # Finishing
    return periods_eps_relevant

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
