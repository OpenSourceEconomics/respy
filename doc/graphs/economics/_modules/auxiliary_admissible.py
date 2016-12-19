import numpy as np
from matplotlib import pyplot as plt

from auxiliary_economics import move_subdirectory

CHOICE_LIST = ['Occupation A', 'Occupation B', 'School', 'Home']
BOUNDS_LIST = ['lower', 'upper']

RSLT = dict()
RSLT[0.000] = [[0.590, 0.590], [0.625, 0.625], [0.635, 0.635], [0.625, 0.625]]
RSLT[0.015] = [[0.560, 0.595], [0.595, 0.630], [0.605, 0.640], [0.595, 0.630]]
RSLT[0.020] = [[0.540, 0.605], [0.575, 0.640], [0.570, 0.650], [0.580, 0.640]]


def _extract_results(level, choice, rslts, which):
    """ Extract results from dictionary.
    """
    idx_choice = CHOICE_LIST.index(choice)
    idx_bounds = BOUNDS_LIST.index(which)

    return rslts[level][idx_choice][idx_bounds]


def get_elements(choice, rslts, selected_grid):
    """  Construct all elements required for the plot of admissible value
    functions.
    """
    lower, upper = [], []

    for which in BOUNDS_LIST:
        rslt = []
        for level in selected_grid:
            rslt += [_extract_results(level, choice, rslts, which)]

        if which in ['upper']:
            upper = np.array(rslt)
        else:
            lower = np.array(rslt)

    increments = upper - lower

    return upper, lower, increments


def plot():
    """ Plot all admissible value functions.
    """

    move_subdirectory()

    rslts = RSLT
    selected_grid = rslts.keys()

    # Fine tuning parameters that allow to shift the box plots.
    ind = np.arange(3) + 0.125
    width = 0.125

    # Initialize clean canvas
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    # Values of Occupation A
    upper, lower, increments = get_elements('Occupation A', rslts, selected_grid)
    ax.bar(ind, lower, width, color='orange', label='Occupation A', alpha=0.7)
    ax.bar(ind, increments, width, color='orange', bottom=lower, hatch='//', alpha=0.7)

    # Values of Occupation B
    upper, lower, increments = get_elements('Occupation B', rslts, selected_grid)
    ax.bar(ind + width * 1, lower, width, color='red', label='Occupation B', alpha=0.7)
    ax.bar(ind + width * 1, increments, width, color='red', bottom=lower, hatch='//', alpha=0.7)

    # Values of School
    upper, lower, increments = get_elements('School', rslts, selected_grid)
    ax.bar(ind + width * 2, lower, width, color='yellow', label='School', alpha=0.7)
    ax.bar(ind + width * 2, increments, width, color='yellow', bottom=lower, hatch='//', alpha=0.7)

    # Values of Home
    upper, lower, increments = get_elements('Home', rslts, selected_grid)
    ax.bar(ind + width * 3, lower, width, color='blue', label='Home', alpha=0.7)
    ax.bar(ind + width * 3, increments, width, color='blue', bottom=lower, hatch='//', alpha=0.7)

    # X Label
    ax.set_xlabel('Level of Ambiguity', fontsize=16)
    ax.set_xlim([0.0, ind[-1] + width * 5])

    ax.set_xticks((ind[0] + width * 2, ind[1] + width * 2, ind[2] + width * 2))
    ax.set_xticklabels(('Absent', 'Low', 'High'))

    # Y Label
    ax.set_ylabel('Value Functions', fontsize=16)
    ax.set_yticklabels([])
    ax.set_ylim([0.5, 0.7])

    # Both Axes
    ax.tick_params(labelsize=16, direction='out', axis='both', top='off', right='off')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=False,
        frameon=False, shadow=False, ncol=4, fontsize=20)

    plt.savefig('admissible_values.png', bbox_inches='tight', format='png')