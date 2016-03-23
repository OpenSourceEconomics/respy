""" This module contains some auxiliary functions to use the
FORTRAN implementation.
"""

# standard library
import numpy as np

import os

# project library
from robupy.shared.auxiliary import replace_missing_values

''' Auxiliary  functions
'''


def _add_results(num_periods, min_idx, request):
    """ Add results to container.
    """

    # Get the maximum number of states. The special treatment is required as
    # it informs about the dimensions of some of the arrays that are
    # processed below.
    max_states_period = int(np.loadtxt('.max_states_period.robufort.dat'))

    os.unlink('.max_states_period.robufort.dat')

    shape = (num_periods, num_periods, num_periods, min_idx, 2)
    mapping_state_idx = _read_date('mapping_state_idx', shape)

    shape = (num_periods,)
    states_number_period = _read_date('states_number_period', shape)

    shape = (num_periods, max_states_period, 4)
    states_all = _read_date('states_all', shape)

    shape = (num_periods, max_states_period, 4)
    periods_payoffs_systematic = _read_date('periods_payoffs_systematic', shape)

    shape = (num_periods, max_states_period, 4)
    periods_payoffs_ex_post = _read_date('periods_payoffs_ex_post', shape)

    shape = (num_periods, max_states_period)
    periods_emax = _read_date('periods_emax', shape)

    # Update class attributes with solution
    args = [periods_payoffs_systematic, periods_payoffs_ex_post,
            None, states_number_period, mapping_state_idx,
            periods_emax, states_all]

    # Finishing
    return args

def _read_date(label, shape):
    """ Read results
    """
    file_ = '.' + label + '.robufort.dat'

    # This special treatment is required as it is crucial for this data
    # to stay of integer type. All other data is transformed to float in
    # the replacement of missing values.
    if label == 'states_number_period':
        data = np.loadtxt(file_, dtype=np.int64)
    else:
        data = replace_missing_values(np.loadtxt(file_))

    data = np.reshape(data, shape)

    # Cleanup
    os.unlink(file_)

    # Finishing
    return data

def _write_robufort_initialization(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
            num_periods, num_points, is_myopic, edu_start, seed_emax,
            is_debug, min_idx, measure, edu_max, delta, level,
            num_draws_prob, num_agents, seed_prob, seed_data, request):
    """ Write out model request to hidden file .model.robufort.ini.
    """


    # Write out to link file
    with open('.model.robufort.ini', 'w') as file_:

        # BASICS
        line = '{0:10d}\n'.format(num_periods)
        file_.write(line)

        line = '{0:15.10f}\n'.format(delta)
        file_.write(line)

        # AMBIGUITY
        line = '{0:15.10f}\n'.format(level)
        file_.write(line)

        if measure == 'kl':
            line = '{0}'.format(measure)
            file_.write(line + '\n')
        else:
            raise NotImplementedError

        # WORK
        for num in [coeffs_a, coeffs_b]:
            line = ' {0:15.10f} {1:15.10f} {2:15.10f} {3:15.10f}  {4:15.10f}' \
                        ' {5:15.10f}\n'.format(*num)
            file_.write(line)

        # EDUCATION
        num = coeffs_edu
        line = ' {0:+15.9f} {1:+15.9f} {2:+15.9f}\n'.format(*num)
        file_.write(line)

        line = '{0:10d} '.format(edu_start)
        file_.write(line)

        line = '{0:10d}\n'.format(edu_max)
        file_.write(line)

        # HOME
        line = ' {0:15.10f}\n'.format(coeffs_home[0])
        file_.write(line)

        # SHOCKS
        for j in range(4):
            line = ' {0:15.5f} {1:15.5f} {2:15.5f} ' \
                   '{3:15.5f}\n'.format(*shocks_cov[j])
            file_.write(line)

        # SOLUTION
        line = '{0:10d}\n'.format(num_draws_emax)
        file_.write(line)

        line = '{0:10d}\n'.format(seed_emax)
        file_.write(line)

        # SIMULATION
        line = '{0:10d}\n'.format(num_agents)
        file_.write(line)

        line = '{0:10d}\n'.format(seed_data)
        file_.write(line)

        # PROGRAM
        line = '{0}'.format(is_debug)
        file_.write(line + '\n')

        # INTERPOLATION
        line = '{0}'.format(is_interpolated)
        file_.write(line + '\n')

        line = '{0:10d}\n'.format(num_points)
        file_.write(line)

        # ESTIMATION
        line = '{0:10d}\n'.format(num_draws_prob)
        file_.write(line)

        line = '{0:10d}\n'.format(seed_prob)
        file_.write(line)

        # Auxiliary
        line = '{0:10d}\n'.format(min_idx)
        file_.write(line)

        line = '{0}'.format(is_ambiguous)
        file_.write(line + '\n')

        line = '{0}'.format(is_deterministic)
        file_.write(line + '\n')

        line = '{0}'.format(is_myopic)
        file_.write(line + '\n')

        # Request
        line = '{0}'.format(request)
        file_.write(line + '\n')
