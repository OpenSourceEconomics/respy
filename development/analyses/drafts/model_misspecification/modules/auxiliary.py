""" Module with some auxiliary methods for the analysis of model
misspecification.
"""

# standard library
import numpy as np

import shutil
import shlex
import os

# project library
from robupy.tests.random_init import print_random_dict

from robupy import solve
from robupy import read


def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    level = args.level

    # Check arguments
    assert (isinstance(level, float))
    assert (level >= 0.00)

    # Finishing
    return level


def solve_true_economy(level):

    os.mkdir('true'), os.chdir('true')
    robupy_obj = read('../model.robupy.ini')
    init_dict = robupy_obj.get_attr('init_dict')
    init_dict['AMBIGUITY']['level'] = level
    # TODO: Remove later
    #init_dict['BASICS']['periods'] = 5

    print_random_dict(init_dict)
    shutil.move('test.robupy.ini', 'model.robupy.ini')
    base_choices = get_baseline('model.robupy.ini')
    os.chdir('../')
    # Finishing
    return base_choices


def get_baseline(name):
    """ Get the baseline distribution.
    """
    # Solve baseline distribution
    robupy_obj = read(name)
    solve(robupy_obj)
    # Get baseline choice distributions
    base_choices = get_period_choices()
    # Finishing
    return base_choices


def criterion_function(point, base_choices):
    """ Get the baseline distribution.
    """
    # Write out the modified baseline initialization.
    robupy_obj = read('test.robupy.ini')
    init_dict = robupy_obj.get_attr('init_dict')
    # Set relevant values
    init_dict['EDUCATION']['int'] = float(point)
    # TODO: Remove later
    #init_dict['BASICS']['periods'] = 5

    # Write to file
    print_random_dict(init_dict)
    # Solve requested model
    robupy_obj = read('test.robupy.ini')
    solve(robupy_obj)
    # Get choice probabilities
    alternative_choices = get_period_choices()
    # Calculate squared mean-deviation of all transition probabilities
    crit = np.mean(np.sum((base_choices[:, :] - alternative_choices[:, :])**2))
    # Finishing
    return crit


def get_period_choices():
    """ Return the share of agents taking a particular decision in each
        of the states for all periods.
    """
    # Auxiliary objects
    file_name, choices = 'data.robupy.info', []
    # Process file
    with open(file_name, 'r') as output_file:
        for line in output_file.readlines():
            # Split lines
            list_ = shlex.split(line)
            # Skip empty lines
            if not list_:
                continue
            # Check if period
            try:
                int(list_[0])
            except ValueError:
                continue
            # Extract information about period decisions
            choices.append([float(x) for x in list_[1:]])
    # Type conversion
    choices = np.array(choices)
    # Finishing
    return choices


def solve_estimated_economy(opt):
    """ Collect information about estimated economy by updating an initialization
    file with the resulting estimate for the intercept and solving the
    resulting economy.
    """
    # Read baseline initialization file
    init_dict = read('test.robupy.ini').get_attr('init_dict')

    # Switch into extra directory to store the results
    os.mkdir('estimated'), os.chdir('estimated')

    # Update initialization file with result from estimation and write to disk
    init_dict['EDUCATION']['int'] = float(opt['x'])
    # TODO: Remove later
    #init_dict['BASICS']['periods'] = 5

    print_random_dict(init_dict)

    # Start solution and estimation of resulting economy
    shutil.move('test.robupy.ini', 'model.robupy.ini')
    solve(read('model.robupy.ini'))

    os.chdir('../')
