""" Module with some auxiliary methods for the analysis of model
misspecification.
"""

# standard library
import shlex
import sys
import os

# scipy library
from scipy.interpolate import interp1d
import numpy as np
try:
    import matplotlib 
    matplotlib.use('Agg')
    import matplotlib.pylab as plt
except ImportError:
    pass

# module-wide variables
ROBUPY_DIR = os.environ['ROBUPY']

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR + '/development/analyses/robust/_scripts')

# _scripts
from _auxiliary import float_to_string
from _auxiliary import get_robupy_obj

# robupy library
from robupy.tests.random_init import print_random_dict
from robupy import solve

# module wide variables
SCALING = 100000.00


def plot_indifference_curve(yvalues, xvalues):
    """ Plot the results from the model misspecification exercise.
    """

    # Set up interpolation
    f = interp1d(xvalues, yvalues, kind='quadratic')
    x_new = np.linspace(0.00, max(xvalues), num=41, endpoint=True)

    # Initialize canvas and basic plot.
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)
    ax.plot(x_new, f(x_new), '-k', color='red', linewidth=5)

    # Both axes
    ax.tick_params(labelsize=18, direction='out', axis='both', top='off',
            right='off')

    # X axis
    ax.set_xlim([0.00, 0.02])
    ax.set_xlabel('Level of Ambiguity', fontsize=16)
    ax.set_xticklabels(('Absent', 'Low', 'High'))
    ax.set_xticks((0.00, 0.01, 0.02))

    # Y axis
    ax.set_ylabel('Intercept', fontsize=16)
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    # Formatting with comma for thousands.
    func = matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.get_yaxis().set_major_formatter(func)

    plt.savefig('rslts/indifference_curve.robupy.png', bbox_inches='tight',
                format='png')


def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    is_recompile = args.is_recompile
    num_procs = args.num_procs
    is_debug = args.is_debug
    level = args.level
    spec = args.spec
    grid = args.grid

    # Check arguments
    assert (isinstance(level, float))
    assert (level >= 0.00)
    assert (is_recompile in [True, False])
    assert (is_debug in [True, False])
    assert (isinstance(num_procs, int))
    assert (num_procs > 0)
    assert (spec in ['one', 'two', 'three'])

    # Check and process information about grid.
    assert (len(grid) == 3)
    grid = float(grid[0]), float(grid[1]), int(grid[2])

    # Finishing
    return level, grid, num_procs, is_recompile, is_debug, spec


def solve_true_economy(init_dict, is_debug):
    """ Solve and store true economy.
    """
    # Prepare directory structure
    os.mkdir('true'), os.chdir('true')
    # Modify initialization dictionary
    if is_debug:
        init_dict['BASICS']['periods'] = 5
    # Solve economy
    solve(get_robupy_obj(init_dict))
    # Get baseline choice distributions
    base_choices = get_period_choices()
    # Store material required for restarting an estimation run
    print_random_dict(init_dict)
    # Return to root directory
    os.chdir('../')
    # Finishing
    return base_choices


def criterion_function(init_dict, base_choices):
    """ Get the baseline distribution.
    """
    # Print out initialization file for debugging.
    print_random_dict(init_dict)
    # Solve requested model
    solve(get_robupy_obj(init_dict))
    # Get choice probabilities
    alternative_choices = get_period_choices()
    # Calculate squared mean-deviation of all transition probabilities
    rslt = np.mean(np.sum((base_choices[:, :] - alternative_choices[:, :])**2))
    # I want to put a particular focus on educational choices.
    rslt += np.mean(np.sum((base_choices[:, 2] - alternative_choices[:, 2])**2))
    # Finishing
    return rslt


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


def get_criterion_function():
    """ Process get value of criterion function.
    """
    with open('indifference_curve.robupy.log', 'r') as output_file:
        for line in output_file.readlines():
            # Split lines
            list_ = shlex.split(line)
            # Skip all irrelevant lines.
            if not len(list_) == 2:
                continue
            if not list_[0] == 'Criterion':
                continue
            # Finishing
            return float(list_[1])


def plot_choice_patterns(choice_probabilities, level):
    """ Function to produce plot for choice patterns.
    """
    labels = ['Home', 'School', 'Occupation A', 'Occupation B']

    deciles = range(40)

    colors = ['blue', 'yellow', 'orange', 'red']

    width = 0.9

    # Plotting
    bottom = [0]*40

    # Initialize plot
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    for i in [3, 2, 1, 0]:

        heights = choice_probabilities[:, i]
        plt.bar(deciles, heights, width, bottom=bottom, color=colors[i])
        bottom = [heights[i] + bottom[i] for i in range(40)]

    # Both Axes
    ax.tick_params(labelsize=16, direction='out', axis='both', top='off',
        right='off')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # X axis
    ax.set_xlabel('Periods', fontsize=16)
    ax.set_xlim([0, 40])

    # Y axis
    ax.set_ylabel('Share of Population', fontsize=16)
    ax.yaxis.get_major_ticks()[0].set_visible(False)
    ax.set_ylim([0, 1])

    # Legend
    plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.10),
        fancybox=False, frameon=False, shadow=False, ncol=4, fontsize=20)

    # Write out to
    plt.savefig('choice_patterns_' + float_to_string(level) + '.robupy.png',
                bbox_inches='tight', format='png')
