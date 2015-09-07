""" This module contains functionality that is shared between the solution
and simulation modules.
"""

# standard library
import numpy as np


def write_robufort_initialization(init_dict):
    """ Write out model request to hidden file .model.robufort.ini.
    """

    with open('.model.robufort.ini', 'w') as file_:

        # BASICS
        line = '{0:10d}\n'.format(init_dict['BASICS']['periods'])
        file_.write(line)

        line = '{0:15.10f}\n'.format(init_dict['BASICS']['delta'])
        file_.write(line)

        # WORK
        for label in ['A', 'B']:
            num = [init_dict[label]['int']] + init_dict[label]['coeff']
            line = ' {0:15.10f} {1:15.10f} {2:15.10f} {3:15.10f}  {4:15.10f}' \
                        ' {5:15.10f}\n'.format(*num)
            file_.write(line)

        # EDUCATION
        num = [init_dict['EDUCATION']['int']] + init_dict['EDUCATION']['coeff']
        line = ' {0:15.10f} {1:15.10f} {2:15.10f}\n'.format(*num)
        file_.write(line)

        line = '{0:10d} '.format(init_dict['EDUCATION']['start'])
        file_.write(line)

        line = '{0:10d}\n'.format(init_dict['EDUCATION']['max'])
        file_.write(line)

        # HOME
        line = '{0:15.10f}\n'.format(init_dict['HOME']['int'])
        file_.write(line)

        # SHOCKS
        shocks = init_dict['SHOCKS']
        for j in range(4):
            line = ' {0:15.5f} {1:15.5f} {2:15.5f} {3:15.5f}\n'.format(*shocks[j])
            file_.write(line)

         # SOLUTION
        line = '{0:10d}\n'.format(init_dict['SOLUTION']['draws'])
        file_.write(line)

        line = '{0:10d}\n'.format(init_dict['SOLUTION']['seed'])
        file_.write(line)

        # SIMULATION
        line = '{0:10d}\n'.format(init_dict['SIMULATION']['agents'])
        file_.write(line)

        line = '{0:10d}\n'.format(init_dict['SIMULATION']['seed'])
        file_.write(line)


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