""" This module contains some auxiliary functions helpful in the
quantification of ambiguity.
"""

# standard library
import numpy as np

from robupy.clsRobupy import RobupyCls

# standard library
import matplotlib.pylab as plt
import matplotlib

matplotlib.use('Agg')


def plot_lifetime_value(rslt):
    """ Plot policy responsivenss.
    """
    # Initialize clean canvas
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    # Collect relevant subset
    response = []
    for key_ in [0.00, 0.01, 0.02]:
        response += [rslt[key_]]

    width = 0.125

    plt.bar(0.5 * width, response[0], width, color='red')

    plt.bar(2.5 * width, response[1], width, color='orange')

    plt.bar(4.5 * width, response[2], width, color='blue')

    # Both Axes
    ax.tick_params(labelsize=16, direction='out', axis='both', top='off',
        right='off')

    # x axis
    ax.set_xlim([0, 6 * width])
    ax.set_xlabel('Level of Ambiguity', fontsize=16)

    # y axis
    ax.set_ylim([330000, 370000])
    ax.yaxis.get_major_ticks()[0].set_visible(False)
    ax.set_ylabel('Expected Lifetime Value', fontsize=16)

    # Formatting of labels
    func = matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.get_yaxis().set_major_formatter(func)

    ax.set_xticks((1.0 * width, 3.0 * width, 5.0 * width))
    ax.set_xticklabels(('0.00', '0.01', '0.02'))

    plt.savefig('rslts/ambiguity_quantification.png', bbox_inches='tight',
                format='png')


def get_robupy_obj(init_dict):
    """ Get the object to pass in the solution method.
    """
    # Initialize and process class
    robupy_obj = RobupyCls()
    robupy_obj.set_attr('init_dict', init_dict)
    robupy_obj.lock()
    # Finishing
    return robupy_obj


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