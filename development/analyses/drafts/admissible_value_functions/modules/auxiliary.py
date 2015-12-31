""" This auxiliary module contains some functions that supports the
investigations of all admissible values.
"""

# standard library
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import shutil
import os

from robupy.python.py.ambiguity import transform_disturbances_ambiguity
from robupy.python.py.auxiliary import simulate_emax
from robupy.clsRobupy import RobupyCls


def _extract_results(level, choice, rslt, which):

    subset = rslt[level]


def plot_admissible_values():
    """
    """

    ind = np.arange(3) + 0.125   # the x locations for the groups
    width = 0.125           # the width of the bars
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    # First we plot all the lowest admissible values.
    lower_values = [1200, 1200, 1200]
    ax.bar(ind, lower_values, width, color='orange',
           label='Occupation A')

    increments = [0, 200, 200]
    ax.bar(ind, increments, width, color='orange',
           bottom=lower_values, hatch='//')

    # Values of Occupation B
    lower_values = [1200, 1200, 1200]
    ax.bar(ind + width * 1, lower_values, width, color='red',
           label='Occupation B')

    increments = [0, 200, 200]
    ax.bar(ind + width * 1, increments, width, color='red',
           bottom=lower_values, hatch='//')

    # Values of Schooling
    lower_values = [1200, 1200, 1200]
    ax.bar(ind + width * 2, lower_values, width, color='yellow',
           label='School')

    increments = [0, 200, 200]
    ax.bar(ind + width * 2, increments, width, color='yellow',
           bottom=lower_values, hatch='//')

    # Values of Home
    lower_values = [1200, 1200, 1200]
    ax.bar(ind + width * 3, lower_values, width, color='blue', label='Home')

    increments = [0, 200, 200]
    ax.bar(ind + width * 3, increments, width, color='green',
           bottom=lower_values, hatch='//')

    # X Label
    ax.set_xlabel('Level of Ambiguity', fontsize=16)
    ax.set_xlim([0.0, ind[-1] + width * 5])

    ax.set_xticks((ind[0] + width * 2, ind[1] + width * 2, ind[2] + width * 2))
    ax.set_xticklabels(('0.00', '0.01', '0.02'))

    # Y Label
    ax.set_ylabel('Expected Future Values', fontsize=16)
    ax.yaxis.get_major_ticks()[0].set_visible(False)
    ax.set_ylim([0.0, 1700])

    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda
        x, p: format(int(x), ',')))

    # Both Axes
    ax.tick_params(labelsize=16, direction='out', axis='both', top='off',
        right='off')

    # Set up te legend nicely formatted.
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
        fancybox=False, frameon=False, shadow=False, ncol=4, fontsize=20)

    # Create a results directory to store the resulting graph.
    if os.path.exists('rslts'):
        shutil.rmtree('rslts')
    os.mkdir('rslts')

    # Store figure as *.png
    plt.savefig('rslts/admissible_values.png', bbox_inches='tight',
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
    levels = args.levels

    # Check arguments
    assert (isinstance(levels, list))
    assert (np.all(levels) >= 0.00)
    assert (is_recompile in [True, False])
    assert (is_debug in [True, False])
    assert (isinstance(num_procs, int))
    assert (num_procs > 0)

    # Finishing
    return levels, is_recompile, is_debug, num_procs


def criterion(x, num_draws, eps_relevant, period, k, payoffs_systematic,
        edu_max, edu_start, mapping_state_idx, states_all, num_periods,
        periods_emax, delta, sign=1):
    """ Simulate expected future value for alternative shock distributions.
    """
    # This is a slightly modified copy of the criterion function in the
    # ambiguity module. The ability to switch the sign was added to allow for
    # maximization as well as minimization.
    assert (sign in [1, -1])

    # Transformation of standard normal deviates to relevant distributions.
    eps_relevant_emax = transform_disturbances_ambiguity(eps_relevant, x)

    # Simulate the expected future value for a given parametrization.
    simulated, _, _ = simulate_emax(num_periods, num_draws, period, k,
                        eps_relevant_emax, payoffs_systematic, edu_max, edu_start,
                        periods_emax, states_all, mapping_state_idx, delta)

    # Finishing
    return sign*simulated


def get_robupy_obj(init_dict):
    """ Get the object to pass in the solution method.
    """
    # Initialize and process class
    robupy_obj = RobupyCls()
    robupy_obj.set_attr('init_dict', init_dict)
    robupy_obj.lock()
    # Finishing
    return robupy_obj
