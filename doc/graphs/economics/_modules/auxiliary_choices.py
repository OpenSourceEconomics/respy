""" This module contains some routines that help in the analysis of policy
responsiveness.
"""

# standard library
import pickle as pkl
import numpy as np
import shlex
import glob
import os

from scipy.interpolate import interp1d
import matplotlib
import shutil
matplotlib.use('Agg')
import matplotlib.pylab as plt

from respy import RespyCls

from auxiliary_economics import float_to_string, get_float_directories

# module-wide variables
OCCUPATIONS = ['Occupation A', 'Occupation B', 'Schooling', 'Home']
MAX_PERIOD = 25
COLORS = ['red', 'orange', 'blue', 'yellow']


def run():
    """ Process results from the models.
    """
    # Cleanup results from a previous run and prepare the directory structure.
    if os.path.exists('rslt'):
        shutil.rmtree('rslt')
    os.mkdir('rslt')

    os.chdir('../grid/rslt')

    # We need to determine the maximum number of periods. The risk-only
    # baseline will always be available.
    os.chdir(float_to_string(0.00))

    num_periods = RespyCls('model.respy.ini').get_attr('num_periods')
    os.chdir('../')

    rslts = dict()
    rslts['final_choices'] = track_final_choices(num_periods)
    rslts['education_period'] = track_schooling_over_time()

    os.chdir('../../effect_ambiguity')

    pkl.dump(rslts, open('rslt/ambiguity_choices.respy.pkl', 'wb'))


def track_final_choices(num_periods):
    """ Track the final choices from the ROBUPY output.
    """
    # Auxiliary objects
    levels = get_levels()

    # Create dictionary with the final shares for varying level of ambiguity.
    shares = dict()
    shares['levels'] = levels
    for occu in OCCUPATIONS:
        shares[occu] = []

    # Iterate over all available ambiguity levels
    for level in levels:
        file_name = float_to_string(level) + '/data.respy.info'
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

                if list_[0] == 'Outcomes':
                    break

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
        file_name = float_to_string(level) + '/data.respy.info'
        with open(file_name, 'r') as output_file:
            for line in output_file.readlines():
                # Split lines
                list_ = shlex.split(line)
                # Skip empty lines
                if not list_:
                    continue

                if list_[0] == 'Outcomes':
                    break

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


def plot(selected_grid):

    rslts = None
    try:
        rslts = pkl.load(open('rslt/ambiguity_choices.respy.pkl', 'rb'))
    except IOError:
        print(' Results not available.')
        SystemExit

    plot_choices_ambiguity(rslts['final_choices'])

    plot_schooling_ambiguity(rslts['education_period'], selected_grid)


def plot_choices_ambiguity(shares_ambiguity):
    """ Plot the choices in the final periods under different levels of ambiguity.
    """

    levels = get_float_directories('../grid/rslt')

    # Initialize plot
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    for i, key_ in enumerate(['Occupation A', 'Occupation B']):

        # Extract relevant data points.
        yvals = shares_ambiguity[key_]
        xvals = levels

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
    ax.set_ylim([0.0, 0.99])
    ax.yaxis.get_major_ticks()[0].set_visible(False)
    ax.set_ylabel('Shares', fontsize=16)

    # Set up legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=False, frameon=False,
        shadow=False, ncol=2, fontsize=20)

    # Write out to
    plt.savefig('rslt/choices_ambiguity.respy.png', bbox_inches='tight',
        format='png')


def plot_schooling_ambiguity(shares_time, selected_grid):
    """ Plot schooling level over time for different levels of ambiguity.
    """
    MAX_PERIOD = 3
    plt.figure(figsize=(12, 8)).add_subplot(111)

    labels = ['Absent', 'Low', 'High']

    # Initialize plot
    ax = plt.figure(figsize=(12,8)).add_subplot(111)

    # Baseline
    for i, label in enumerate(selected_grid):

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

    file_name = 'rslt/schooling_ambiguity.respy.png'

    # Write out to
    plt.savefig(file_name, bbox_inches='tight', format='png')


