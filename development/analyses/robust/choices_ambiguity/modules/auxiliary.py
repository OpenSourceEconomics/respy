""" This module contains some routines that help in the analysis of policy
responsiveness.
"""

# standard library
import shlex
import glob
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
sys.path.insert(0, ROBUPY_DIR+ '/development/analyses/robust/_scripts')
sys.path.insert(0, ROBUPY_DIR)

# _scripts
from _auxiliary import float_to_string

# module-wide variables
OCCUPATIONS = ['Occupation A', 'Occupation B', 'Schooling', 'Home']
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
    num_procs = args.num_procs
    is_recompile = args.is_recompile
    is_debug = args.is_debug
    levels = args.levels
    spec = args.spec

    # Check arguments
    assert (num_procs > 0)
    assert (spec in ['one', 'two', 'three'])
    assert (isinstance(levels, list))
    assert (np.all(levels) >= 0.00)

    # Finishing
    return levels, num_procs, is_recompile, is_debug, spec


def get_results(init_dict, is_debug):
    """ Process results from the models.
    """

    os.chdir('rslts')

    rslts = dict()

    rslts['final_choices'] = track_final_choices(init_dict, is_debug)

    rslts['education_period'] = track_schooling_over_time()

    os.chdir('../')

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
        file_name = float_to_string(level) + '/data.robupy.info'
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
        file_name = float_to_string(level) + '/data.robupy.info'
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
        levels += [float(level)]

    # Finishing
    return sorted(levels)



""" Plotting function
"""


def plot_choices_ambiguity(shares_ambiguity):
    """ Plot the choices in the final periods under different levels of
    ambiguity.
    """

    levels = [0.0, 0.0033, 0.0142]

    # Initialize plot
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    for i, key_ in enumerate(['Occupation A', 'Occupation B']):

        # Extract relevant data points.
        xvals = levels
        yvals = shares_ambiguity[key_]

        # Set up interpolation
        f = interp1d(xvals, yvals, kind='quadratic')
        x_new = np.linspace(0.00, max(levels), num=41, endpoint=True)

        # Plot interpolation results
        ax.plot(x_new, f(x_new), linewidth=5, label=key_, color=COLORS[i],
                alpha=0.8)

    # Both axes
    ax.tick_params(labelsize=18, direction='out', axis='both', top='off',
            right='off')

    # x axis
    ax.set_xlim([levels[0], levels[-1]])
    ax.set_xlabel('Level of Ambiguity', fontsize=16)
    ax.set_xticks(levels)
    ax.set_xticklabels(('Absent', 'Low', 'High'))

    # y axis
    ax.set_ylim([0.3, 0.65])
    ax.yaxis.get_major_ticks()[0].set_visible(False)
    ax.set_ylabel('Shares', fontsize=16)

    # Set up legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
            fancybox=False, frameon=False, shadow=False,
            ncol=2, fontsize=20)

    # Write out to
    plt.savefig('rslts/choices_ambiguity.robupy.png', bbox_inches='tight',
                format='png')


def plot_schooling_ambiguity(shares_time):
    """ Plot schooling level over time for different levels of ambiguity.
    """
    plt.figure(figsize=(12, 8)).add_subplot(111)

    labels = ['Absent', 'Low', 'High']

    # Initialize plot
    ax = plt.figure(figsize=(12,8)).add_subplot(111)

    # Baseline
    for i, label in enumerate([0.0, 0.0033, 0.0142]):

            yvalues = range(1 + 15, MAX_PERIOD + 1 + 15)
            xvalues = shares_time[label]['Schooling'][:MAX_PERIOD]
            ax.plot(yvalues, xvalues, label=labels[i], linewidth=5,
                    color=COLORS[i])

    # Both axes
    ax.tick_params(labelsize=18, direction='out', axis='both', top='off',
                   right='off')

    # Remove first element on y-axis
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    ax.set_xlim([1 + 15, MAX_PERIOD + 15]), ax.set_ylim([0, 0.60])

    # labels
    ax.set_xlabel('Age', fontsize=16)
    ax.set_ylabel('Share in School', fontsize=16)

    # Set up legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
        fancybox=False, frameon=False, shadow=False,
        ncol=3, fontsize=20)

    file_name = 'schooling_ambiguity.robupy.png'

    # Write out to
    plt.savefig('rslts/' + file_name, bbox_inches='tight', format='png')