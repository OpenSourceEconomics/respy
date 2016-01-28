""" Module with some auxiliary methods for the analysis of model
misspecification.
"""

# standard library
import pickle as pkl

import shutil
import shlex
import glob
import sys
import os

# scipy library
from scipy.interpolate import interp1d
import numpy as np
try:
    import matplotlib.pylab as plt
    import matplotlib
except ImportError:
    pass

# module-wide variables
ROBUPY_DIR = os.environ['ROBUPY']
SCALING = 10000.00

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR + '/development/analyses/robust/_scripts')

# _scripts
from _auxiliary import check_indifference
from _auxiliary import get_robupy_obj

# project library
from robupy.tests.random_init import print_random_dict
from robupy import solve


def store_results(rslt):
    """ Store results for further processing.
    """
    # Auxiliary objects.
    levels = rslt.keys()

    with open('rslts/model_misspecification.robupy.log', 'w') as out_file:

        # Heading
        fmt = ' {0:<15}{1:<15}\n\n'
        args = ('Level', 'Intercept')
        out_file.write(fmt.format(*args))

        # Formatting for all remaining output.
        for level in levels:
            fmt = ' {0:<15.4f}{1:<15.3f}\n'
            args = (level, rslt[level])

            # Write out optimal information.
            out_file.write(fmt.format(*args))

        out_file.write('\n')

    pkl.dump(rslt, open('rslts/model_misspecification.robupy.pkl', 'wb'))


def cleanup_directory(name):
    """ Cleanup directory for a restart of the estimation.
    """
    os.chdir(name)

    candidates = glob.glob('*')
    for candidate in candidates:
        if 'true' in candidate:
            continue
        _remove(candidate)
    os.chdir('../')


def _remove(names):
    """ Remove files or directories.
    """
    if not isinstance(names, list):
        names = [names]

    for name in names:
        try:
            os.remove(name)
        except IsADirectoryError:
            shutil.rmtree(name)


def plot_model_misspecification(yvalues, xvalues):
    """ Plot the results from the model misspecification exercise.
    """

    # Set up interpolation
    f = interp1d(xvalues, yvalues, kind='quadratic')
    x_new = np.linspace(0.00, 0.02, num=41, endpoint=True)

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

    plt.savefig('rslts/model_misspecification.robupy.png', bbox_inches='tight',
                format='png')


def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    is_recompile = args.is_recompile
    is_restart = args.is_restart
    num_procs = args.num_procs
    is_debug = args.is_debug
    spec = args.spec

    # Check arguments
    assert (is_restart in [True, False])
    assert (is_recompile in [True, False])
    assert (is_debug in [True, False])
    assert (isinstance(num_procs, int))
    assert (num_procs > 0)
    assert (spec in ['one', 'two', 'three'])

    # If not a debugging run, then the indifference points need to be available.
    if not is_debug:
        check_indifference()

    # Finishing
    return num_procs, is_recompile, is_debug, is_restart, spec


def solve_true_economy(init_dict, is_debug):
    """ Solve and store true economy.
    """
    # Prepare directory structure
    os.mkdir('true'), os.chdir('true')
    # Modify initialization dictionary
    # Solve economy
    solve(get_robupy_obj(init_dict))
    # Get baseline choice distributions
    base_choices = get_period_choices()
    # Store material required for restarting an estimation run
    pkl.dump(base_choices, open('base_choices.pkl', 'wb'))
    print_random_dict(init_dict)
    # Return to root directory
    os.chdir('../')


def criterion_function(point, base_choices, init_dict, is_debug):
    """ Get the baseline distribution.
    """
    # Set relevant values
    init_dict['EDUCATION']['int'] = float(point)*SCALING
    # Solve requested model
    solve(get_robupy_obj(init_dict))
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


def solve_estimated_economy(opt, init_dict):
    """ Collect information about estimated economy by updating an initialization
    file with the resulting estimate for the intercept and solving the
    resulting economy.
    """
    # Switch into extra directory to store the results
    os.mkdir('estimated'), os.chdir('estimated')
    # Update initialization file with result from estimation and write to disk
    init_dict['EDUCATION']['int'] = float(opt['x']) * SCALING
    # Solve the basic economy
    robupy_obj = solve(get_robupy_obj(init_dict))
    # Extract result
    init_dict = robupy_obj.get_attr('init_dict')
    rslt = init_dict['EDUCATION']['int']
    # Store some material for debugging purposes
    print_random_dict(init_dict)
    # Return to root dictionary
    os.chdir('../')
    # Finishing
    return rslt
