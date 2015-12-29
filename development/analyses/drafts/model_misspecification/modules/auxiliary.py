""" Module with some auxiliary methods for the analysis of model
misspecification.
"""

# standard library
import matplotlib.pylab as plt
import pickle as pkl
import numpy as np

import shutil
import shlex
import glob
import os

# project library
from robupy.clsRobupy import RobupyCls
from robupy import solve


def plot_model_misspecification(yvalues, xvalues):
    """ Plot the results from the model misspecification exercise.
    """

    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    ax.plot(yvalues, xvalues, '-k', color='red', linewidth=5)

    # X axis
    ax.set_xlabel('Level of Ambiguity', fontsize=16)

    # Y axis
    ax.set_ylabel('Intercept', fontsize=16)
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    #bar_width = 0.35
    #plt.xticks(np.arange(3) + bar_width, ('0.00', '0.01', '0.02'))

    plt.savefig('rslts/model_misspecification.png', bbox_inches='tight',
                format='png')



def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    is_restart = args.is_restart
    num_procs = args.num_procs
    is_debug = args.is_debug
    levels = args.levels

    # Check arguments
    assert (isinstance(levels, list))
    assert (np.all(levels) >= 0.00)
    assert (is_restart in [True, False])
    assert (is_debug in [True, False])
    assert (isinstance(num_procs, int))
    assert (num_procs > 0)

    # Check for restart material
    if is_restart:
        for level in levels:
            assert (os.path.exists('%03.3f/true/base_choices.pkl' % level))

    # Finishing
    return levels, num_procs, is_restart, is_debug


def prepare_directory(is_restart, name):
    """ Prepare directory structure depending whether a fresh start or we are
    building on existing results.
    """
    if not is_restart:
        if os.path.exists(name):
            shutil.rmtree(name)
        os.mkdir(name)
    else:
        os.chdir(name)
        for obj in glob.glob('*'):
            # Retain results from previous run
            if 'true' in obj: continue
            # Delete files and directories
            try:
                shutil.rmtree(obj)
            except OSError:
                os.unlink(obj)
        os.chdir('../')


def solve_true_economy(level, init_dict, is_debug):
    """ Solve and store true economy.
    """
    # Prepare directory structure
    os.mkdir('true'), os.chdir('true')
    # Modify initialization dictionary
    init_dict['AMBIGUITY']['level'] = level
    if is_debug:
        init_dict['BASICS']['periods'] = 5
    # Solve economy
    solve(_get_robupy_obj(init_dict))
    # Get baseline choice distributions
    base_choices = get_period_choices()
    # Store material required for restarting an estimation run
    pkl.dump(base_choices, open('base_choices.pkl', 'wb'))
    # Return to root directory
    os.chdir('../')


def criterion_function(point, base_choices, init_dict, is_debug):
    """ Get the baseline distribution.
    """
    # Auxiliary objects
    scaling = 100000.00
    # Set relevant values
    init_dict['EDUCATION']['int'] = float(point)*scaling
    if is_debug:
        init_dict['BASICS']['periods'] = 5
    # Solve requested model
    solve(_get_robupy_obj(init_dict))
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


def solve_estimated_economy(opt, init_dict, is_debug):
    """ Collect information about estimated economy by updating an initialization
    file with the resulting estimate for the intercept and solving the
    resulting economy.
    """
    # Switch into extra directory to store the results
    os.mkdir('estimated'), os.chdir('estimated')
    # Update initialization file with result from estimation and write to disk
    init_dict['EDUCATION']['int'] = float(opt['x'])
    if is_debug:
        init_dict['BASICS']['periods'] = 5
    # Solve the basic economy
    solve(_get_robupy_obj(init_dict))
    # Return to root dictionary
    os.chdir('../')


def _get_robupy_obj(init_dict):
    """ Get the object to pass in the solution method.
    """
    # Initialize and process class
    robupy_obj = RobupyCls()
    robupy_obj.set_attr('init_dict', init_dict)
    robupy_obj.lock()
    # Finishing
    return robupy_obj
