""" This module contains some auxiliary functions for the illustration of the
modelling tradeoffs.
"""

# standard library
from multiprocessing import Array
from multiprocessing import Pool

RSLT = Array('d', 100)

from functools import partial
import numpy as np

import shutil
import shlex
import glob
import os

# project library
from robupy import *
from robupy.tests.random_init import print_random_dict


def cleanup():
    """ This function cleans the baseline directory to start with a cleaned
    slate.
    """
    for candidate in glob.glob('*'):
        # Skip required files
        if 'driver' in candidate:
            continue
        if 'model.robupy.ini' in candidate:
            continue
        if 'modules' in candidate:
            continue
        if 'create' in candidate:
            continue
        # Remove files
        try:
            os.remove(candidate)
        except IsADirectoryError:
            shutil.rmtree(candidate)


def write_logging(grid_points, ambiguity, rslt):
    """ Write out some information to monitor the construction of the figure.
    """
    # Auxiliary objects
    parameter = str(grid_points[np.argmin(rslt)])
    string = '''{0[0]:10.4f}     {0[1]:10.4f}\n'''

    # Write to file
    with open('indifference.robupy.log', 'w') as file_:

        file_.write('\n Ambiguity ' + str(ambiguity) + '  Parameter ' + parameter +
                    '\n\n')

        file_.write(' Parameter      Criterion \n\n')

        for i, point in enumerate(grid_points):
            file_.write(string.format([point, rslt[i]]))


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


def criterion_function(base_choices, idx, point):
    """ The criterion function. It compares the moments for the educational
    choices only.
    """
    # Process baseline file
    robupy_obj = read('test.robupy.ini')
    init_dict = robupy_obj.get_attr('init_dict')

    # Set relevant values
    init_dict['A']['int'] = float(point)

    # Write initialization file
    print_random_dict(init_dict)

    # Solve requested model
    robupy_obj = read('test.robupy.ini')

    solve(robupy_obj)

    # Get choice probabilities
    alternative_choices = get_period_choices()

    # Calculate squared mean-deviation of transition probabilites
    crit = np.mean(np.sum((base_choices[:, 3] - alternative_choices[:, 3])**2))

    RSLT[idx] = crit

    # Finishing
    return crit


def solve_ambiguous_economy(base_choices, COST_GRID, INNER_NUM_PROCS,
                            ambiguity):

    # Create explicit directory
    name = str(ambiguity)
    grid_points = COST_GRID[ambiguity]

    os.mkdir(name), os.chdir(name)

    shutil.copy('../model.robupy.ini', 'model.robupy.ini')

    # Write out the modified baseline initialization.
    robupy_obj = read('model.robupy.ini')
    init_dict = robupy_obj.get_attr('init_dict')

    # Set relevant values
    init_dict['AMBIGUITY']['level'] = ambiguity

    # Write to file
    print_random_dict(init_dict)

    # Evaluate all admissible values in parallel
    p = Pool(INNER_NUM_PROCS)

    distributed_function = partial(criterion_function, base_choices, grid_points)

    p.map(distributed_function, range(len(grid_points)))

    rslt = [9.21, 9.22, 9.23, 9.24, 9.25]

    # Write out information of evaluations
    write_logging(grid_points, ambiguity, rslt)

    # Return to root directory
    os.chdir('../')