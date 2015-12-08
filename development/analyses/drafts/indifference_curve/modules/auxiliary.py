""" This module contains some auxiliary functions for the illustration of the
modelling tradeoffs.
"""

# standard library
import numpy as np

import random
import string
import shutil
import shlex
import os

# project library
from robupy.tests.random_init import print_random_dict
from robupy import *


def get_random_string(n=10):
    """ Generate a random string of length n.
    """

    return ''.join(random.choice(string.ascii_uppercase +
                string.digits) for _ in range(n))


def write_logging(AMBIGUITY_GRID, COST_GRID, final):
    """ Write out some information to monitor the construction of the figure.
    """
    # Auxiliary objects
    str_ = '''{0[0]:10.4f}     {0[1]:10.4f}\n'''
    # Write to file
    with open('indifference.robupy.log', 'w') as file_:
        for i, ambi in enumerate(AMBIGUITY_GRID):
            # Determine optimal point
            parameter = str(COST_GRID[ambi][np.argmin(final[i,:])])
            # Structure output
            file_.write('\n Ambiguity ' + str(ambi) + '  Parameter ' +
                        parameter + '\n\n')
            file_.write('     Point      Criterion \n\n')
            # Provide additional information about all checked values.
            for j, point in enumerate(COST_GRID[ambi]):
                file_.write(str_.format([point, final[i, j]]))
            file_.write('\n')


def get_period_choices():
    """ Return the share of agents taking a particular decision in each
        of the states for all periods.
    """
    # Auxiliary objects
    file_name = 'data.robupy.info'
    choices = []

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


def get_baseline():
    """ Get the baseline distribution.
    """
    robupy_obj = read('model.robupy.ini')

    solve(robupy_obj)

    base_choices = get_period_choices()

    return base_choices


def pair_evaluation(base_choices, x):

    # Distribute input arguments
    ambi, point = x

    # Auxiliary objects
    name = get_random_string()

    # Move to random directory
    os.mkdir(name), os.chdir(name)

    shutil.copy('../model.robupy.ini', 'model.robupy.ini')

    # Write out the modified baseline initialization.
    robupy_obj = read('model.robupy.ini')
    init_dict = robupy_obj.get_attr('init_dict')

    # Set relevant values
    init_dict['AMBIGUITY']['level'] = ambi
    init_dict['A']['int'] = point

    # Write to file
    print_random_dict(init_dict)

    # Solve requested model
    robupy_obj = read('test.robupy.ini')

    solve(robupy_obj)

    # Get choice probabilities
    alternative_choices = get_period_choices()

    # Calculate squared mean-deviation of transition probabilites
    crit = np.mean(np.sum((base_choices[:, 3] - alternative_choices[:, 3])**2))

    os.chdir('../')

    # Cleanup
    shutil.rmtree(name)

    # Finishing
    return crit
