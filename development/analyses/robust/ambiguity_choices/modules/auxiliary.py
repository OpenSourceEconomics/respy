""" This module contains some routines that help in the analysis of policy
responsiveness.
"""

# standard library
import matplotlib
matplotlib.use('Agg')

import matplotlib.pylab as plt
import pickle as pkl

from stat import S_ISDIR

import shlex
import glob
import sys
import os

from robupy.clsRobupy import RobupyCls

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'])

# module-wide variables
OCCUPATIONS = ['Occupation A', 'Occupation B', 'Schooling', 'Home']
LABELS_SUBSET = ['0.000', '0.010', '0.020']
MAX_PERIOD = 25
COLORS = ['red', 'orange', 'blue', 'yellow']

""" Auxiliary functions
"""

def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    num_procs, grid = args.num_procs,  args.grid
    is_recompile = args.is_recompile
    is_debug = args.is_debug

    # Check arguments
    assert (num_procs > 0)

    if grid != 0:
        assert (len(grid) == 3)
        assert (grid[2] > 0.0)
        assert (grid[0] < grid[1]) or ((grid[0] == 0) and (grid[1] == 0))

    # Finishing
    return num_procs, grid, is_recompile, is_debug


def get_robupy_obj(init_dict):
    """ Get the object to pass in the solution method.
    """
    # Initialize and process class
    robupy_obj = RobupyCls()
    robupy_obj.set_attr('init_dict', init_dict)
    robupy_obj.lock()
    # Finishing
    return robupy_obj


def get_results(init_dict, is_debug):
    """ Process results from the models.
    """

    rslts = dict()

    rslts['final_choices'] = track_final_choices(init_dict, is_debug)

    rslts['education_period'] = track_schooling_over_time()

    # Finishing
    return rslts


def track_final_choices(init_dict, is_debug):
    """ Track the final choices from the ROBUPY output.
    """
    # Auxiliary objects
    levels = get_levels()

    # Process benchmark file
    num_periods = init_dict['BASICS']['periods']

    # Restrict number of periods for debugging purposes.
    if is_debug:
        num_periods = 3

    # Create dictionary with the final shares for varying level of ambiguity.
    shares = dict()
    shares['levels'] = levels
    for occu in OCCUPATIONS:
        shares[occu] = []

    # Iterate over all available ambiguity levels
    for level in levels:
        file_name = level + '/data.robupy.info'
        with open(file_name, 'r') as output_file:
            for line in output_file.readlines():
                # Split lines
                list_ = shlex.split(line)
                # Skip empty lines
                if not list_:
                    continue
                # Extract shares
                if str(num_periods) in list_[0]:
                    for i, occu in enumerate(OCCUPATIONS):
                        shares[occu] += [float(list_[i + 1])]

    # Finishing
    return shares


def track_schooling_over_time():
    """ Create dictionary which contains the simulated shares over time for
    varying levels of ambiguity.
    """
    # Auxiliary objects
    levels = get_levels()

    # Iterate over results
    shares = dict()
    shares['levels'] = levels
    for level in levels:
        # Construct dictionary
        shares[level] = dict()
        for choice in OCCUPATIONS:
            shares[level][choice] = []
        # Process results
        file_name = level + '/data.robupy.info'
        with open(file_name, 'r') as output_file:
            for line in output_file.readlines():
                # Split lines
                list_ = shlex.split(line)
                # Check relevance
                try:
                    int(list_[0])
                except ValueError:
                    continue
                except IndexError:
                    continue
                # Process line
                for i, occu in enumerate(OCCUPATIONS):
                    shares[level][occu] += [float(list_[i + 1])]

    # Finishing
    return shares


def get_levels():
    """ Infer ambiguity levels from directory structure.
    """

    levels = []

    for level in glob.glob('*'):
        # Check if directory name can be transformed into a float.
        try:
            float(level)
        except ValueError:
            continue
        # Collect levels
        levels += [level]

    # Finishing
    return sorted(levels)


def isdir(path, sftp):
    """ Check whether path is a directory.
    """
    try:
        return S_ISDIR(sftp.stat(path).st_mode)
    except IOError:
        return False


def formatting(arg):
    """ Formatting of float input to standardized string.
    """
    return '{0:0.3f}'.format(arg)

""" Plotting function
"""


def plot_choices_ambiguity(shares_ambiguity):
    """ Plot the choices in the final periods under different levels of
    ambiguity.
    """
    # Extract information
    levels = shares_ambiguity['levels']

    # Initialize plot
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    # Draw lines
    # TODO: ECON edit
    # for i, key_ in enumerate(OCCUPATIONS):
    for i, key_ in enumerate(['Occupation A', 'Occupation B']):
        ax.plot(levels, shares_ambiguity[key_], linewidth=5, label=key_,
            color=COLORS[i])


    # Both axes
    ax.tick_params(labelsize=18, direction='out', axis='both', top='off',
            right='off')

    # x axis
    ax.set_xlim([float(levels[0]), float(levels[-1])])
    ax.set_xlabel('Level of Ambiguity', fontsize=16)
    ax.set_xticks((0.00, 0.01, 0.02))
    ax.set_xticklabels(('Absent', 'Low', 'High'))

    # y axis
    ax.set_ylim([0, 1])
    ax.yaxis.get_major_ticks()[0].set_visible(False)
    ax.set_ylabel('Shares', fontsize=16)

    # Set up legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
            fancybox=False, frameon=False, shadow=False,
            ncol=2, fontsize=20)

    # Write out to
    plt.savefig('rslts/choices_ambiguity.png', bbox_inches='tight',
                format='png')


def plot_schooling_ambiguity(shares_time):
    """ Plot schooling level over time for different levels of ambiguity.
    """
    plt.figure(figsize=(12, 8)).add_subplot(111)

    # TODO: Special labels for Harris Talk
    if False:
        theta = [r'$\theta$', r'$\theta^{\prime}$', r'$\theta^{\\prime\prime}$']
    elif True:
        theta = ['Absent', 'Low', 'High']
    else:
        theta = [r'$\theta = 0.00$', r'$\theta^{\prime} = 0.01$', r'$\theta^{\prime\prime}  = 0.02$']

    # Initialize plot
    for choice in ['Schooling']:

        # Initialize canvas
        ax = plt.figure(figsize=(12,8)).add_subplot(111)

        # Baseline
        for i, label in enumerate(LABELS_SUBSET):
            # TODO: Special labels for Harris Talk
            if False:
                yvalues, xvalues = range(1, MAX_PERIOD + 1), shares_time[label][choice][
                                         :MAX_PERIOD]
            else:
                yvalues, xvalues = range(1 + 15, MAX_PERIOD + 1 + 15), \
                                   shares_time[label][choice][
                                         :MAX_PERIOD]


            ax.plot(yvalues, xvalues, label=theta[i], linewidth=5,
                    color=COLORS[i])

        # Both axes
        ax.tick_params(labelsize=18, direction='out', axis='both', top='off',
            right='off')

        # Remove first element on y-axis
        ax.yaxis.get_major_ticks()[0].set_visible(False)

        # TODO: Special labels for Harris Talk
        if False:
            ax.set_xlim([1, MAX_PERIOD]), ax.set_ylim([0, 0.60])
        else:
            ax.set_xlim([1 + 15, MAX_PERIOD + 15]), ax.set_ylim([0, 0.60])

        # labels
        ax.set_xlabel('Age', fontsize=16)
        ax.set_ylabel('Share in School', fontsize=16)

        # Set up legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
            fancybox=False, frameon=False, shadow=False,
            ncol=len(LABELS_SUBSET), fontsize=20)

        file_name = choice.replace(' ', '_').lower() + '_ambiguity.png'

        # Write out to
        plt.savefig('rslts/' + file_name, bbox_inches='tight', format='png')