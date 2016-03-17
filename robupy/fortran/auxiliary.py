""" This module contains some auxiliary functions to use the
FORTRAN implementation.
"""

# standard library
import numpy as np

import os

# module-wide variables
PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))

# project library
from robupy.auxiliary import replace_missing_values

''' Auxiliary  functions
'''


def _add_results(robupy_obj, request):
    """ Add results to container.
    """
    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')

    min_idx = robupy_obj.get_attr('min_idx')

    # Get the maximum number of states. The special treatment is required as
    # it informs about the dimensions of some of the arrays that are
    # processed below.
    max_states_period = int(np.loadtxt('.max_states_period.robufort.dat'))

    os.unlink('.max_states_period.robufort.dat')

    # Labels for objects
    labels = []

    labels += ['mapping_state_idx']

    labels += ['states_number_period']

    labels += ['states_all']

    labels += ['periods_payoffs_systematic']

    labels += ['periods_payoffs_ex_post']

    labels += ['periods_emax']

    # Shapes for the final arrays
    shapes = []

    shapes += [(num_periods, num_periods, num_periods, min_idx, 2)]

    shapes += [(num_periods,)]

    shapes += [(num_periods, max_states_period, 4)]

    shapes += [(num_periods, max_states_period, 4)]

    shapes += [(num_periods, max_states_period, 4)]

    shapes += [(num_periods, max_states_period)]

    # Add objects to class instance
    robupy_obj.unlock()

    for i in range(6):

        label, shape = labels[i], shapes[i]

        file_ = '.' + label + '.robufort.dat'

        # This special treatment is required as it is crucial for this data
        # to stay of integer type. All other data is transformed to float in
        # the replacement of missing values.
        if label == 'states_number_period':
            data = np.loadtxt(file_, dtype=np.int64)
        else:
            data = replace_missing_values(np.loadtxt(file_))

        data = np.reshape(data, shape)

        robupy_obj.set_attr(label, data)

        os.unlink(file_)

    robupy_obj.lock()

    # Read in evaluation of criterion function
    eval_ = None
    if request == 'evaluate':
        eval_ = float(np.genfromtxt('.eval.robufort.dat'))
        os.unlink('.eval.robufort.dat')

    # Finishing
    return robupy_obj, eval_


def _write_robufort_initialization(robupy_obj, request):
    """ Write out model request to hidden file .model.robufort.ini.
    """
    # Distribute class attributes
    model_paras = robupy_obj.get_attr('model_paras')

    # Auxiliary objects
    is_deterministic = robupy_obj.get_attr('is_deterministic')

    is_interpolated = robupy_obj.get_attr('is_interpolated')

    num_draws_prob = robupy_obj.get_attr('num_draws_prob')

    num_draws_emax = robupy_obj.get_attr('num_draws_emax')

    is_ambiguous = robupy_obj.get_attr('is_ambiguous')

    num_periods = robupy_obj.get_attr('num_periods')

    num_points = robupy_obj.get_attr('num_points')

    num_agents = robupy_obj.get_attr('num_agents')

    seed_prob = robupy_obj.get_attr('seed_prob')

    seed_data = robupy_obj.get_attr('seed_data')

    is_myopic = robupy_obj.get_attr('is_myopic')

    edu_start = robupy_obj.get_attr('edu_start')

    seed_emax = robupy_obj.get_attr('seed_emax')

    is_debug = robupy_obj.get_attr('is_debug')

    min_idx = robupy_obj.get_attr('min_idx')

    measure = robupy_obj.get_attr('measure')

    edu_max = robupy_obj.get_attr('edu_max')

    delta = robupy_obj.get_attr('delta')

    level = robupy_obj.get_attr('level')

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
        for num in [model_paras['coeffs_a'], model_paras['coeffs_b']]:
            line = ' {0:15.10f} {1:15.10f} {2:15.10f} {3:15.10f}  {4:15.10f}' \
                        ' {5:15.10f}\n'.format(*num)
            file_.write(line)

        # EDUCATION
        num = model_paras['coeffs_edu']
        line = ' {0:+15.9f} {1:+15.9f} {2:+15.9f}\n'.format(*num)
        file_.write(line)

        line = '{0:10d} '.format(edu_start)
        file_.write(line)

        line = '{0:10d}\n'.format(edu_max)
        file_.write(line)

        # HOME
        line = ' {0:15.10f}\n'.format(model_paras['coeffs_home'][0])
        file_.write(line)

        # SHOCKS
        shocks = model_paras['shocks']
        for j in range(4):
            line = ' {0:15.5f} {1:15.5f} {2:15.5f} ' \
                   '{3:15.5f}\n'.format(*shocks[j])
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
