""" This module contains some auxiliary functions for the illustration of the
modelling trade-offs.
"""

# standard library
import numpy as np

import random
import string
import shutil
import shlex
import os

# project library
from robupy.clsRobupy import RobupyCls
from robupy import solve


def check_grid(AMBIGUITY_GRID, COST_GRID):
    """ Check the manual specification of the grid.
    """
    # Auxiliary objects
    num_eval_points = len(COST_GRID[0])
    # Check that points for all levels of ambiguity levels are defined
    assert (set(COST_GRID.keys()) == set(AMBIGUITY_GRID))
    # Check that the same number of points is requested. This ensures the
    # symmetry of the evaluation grid for the parallelization request.
    for key_ in AMBIGUITY_GRID:
        assert (len(COST_GRID[key_]) == num_eval_points)
    # Make sure that there are no duplicates in the grid.
    for key_ in AMBIGUITY_GRID:
        assert (len(COST_GRID[key_]) == len(set(COST_GRID[key_])))


def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    num_procs = args.num_procs
    is_debug = args.is_debug

    # Check arguments
    assert (num_procs > 0)
    assert (is_debug in [True, False])

    # Finishing
    return num_procs, is_debug


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


def get_baseline(init_dict):
    """ Get the baseline distribution.
    """
    name = get_random_string()
    # Move to random directory and get baseline model specification
    os.mkdir(name), os.chdir(name)
    solve(_get_robupy_obj(init_dict))
    # Get baseline choice distributions
    base_choices = get_period_choices()
    # Cleanup
    os.chdir('../'), shutil.rmtree(name)
    # Finishing
    return base_choices


def pair_evaluation(init_dict, base_choices, x):
    """ Evaluate criterion function for alternative pairs of ambiguity and
    point.
    """
    # Distribute input arguments
    ambi, point = x
    # Auxiliary objects
    name = get_random_string()
    # Move to random directory and get baseline model specification
    os.mkdir(name), os.chdir(name)
    # Set relevant values
    init_dict['AMBIGUITY']['level'] = ambi
    # Set relevant values
    init_dict['EDUCATION']['int'] = float(point)
    # Solve requested model
    solve(_get_robupy_obj(init_dict))
    # Get choice probabilities
    alternative_choices = get_period_choices()
    # Calculate squared mean-deviation of transition probabilities
    crit = np.mean(np.sum((base_choices[:, :] - alternative_choices[:, :])**2))
    # Cleanup
    os.chdir('../'), shutil.rmtree(name)
    # Finishing
    return crit


def _get_robupy_obj(init_dict):
    """ Get the object to pass in the solution method.
    """
    # Initialize and process class
    robupy_obj = RobupyCls()
    robupy_obj.set_attr('init_dict', init_dict)
    robupy_obj.lock()
    # Finishing
    return robupy_obj