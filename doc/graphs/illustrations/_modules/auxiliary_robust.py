import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import numpy as np

from scipy.interpolate import interp1d

from auxiliary_economics import  move_subdirectory

def plot():

    move_subdirectory()

    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    x_values = [0.00, 0.33, 0.66, 1.00]
    y_values = [0.95, 0.70, 0.40, 0.00]

    f = interp1d(x_values, y_values, kind='quadratic')
    x_grid = np.linspace(0, 1, num=41, endpoint=True)
    ax.plot(x_grid, f(x_grid), label='Optimal', linewidth=5, color='red')


    x_values = [0.00, 0.33, 0.66, 1.00]
    y_values = [0.80, 0.70, 0.50, 0.20]

    f = interp1d(x_values, y_values, kind='quadratic')
    x_grid = np.linspace(0, 1, num=41, endpoint=True)
    ax.plot(x_grid, f(x_grid), label='Robust', linewidth=5, color='black')

    ax.set_xticks((0.2, 0.5, 0.8))
    ax.set_xticklabels(('Low', 'Medium', 'High'))

    # Both axes
    ax.tick_params(labelsize=18, direction='out', axis='both', top='off', right='off')

    # Remove first element on y-axis
    ax.yaxis.get_major_ticks()[0].set_visible(False)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_yticklabels([])

    # # labels
    ax.set_xlabel('Worst-Case Model', fontsize=16)
    ax.set_ylabel('Lifetime Utility', fontsize=16)
    # Both Axes
    ax.tick_params(labelsize=16, direction='out', axis='both', top='off',
        right='off')
    # Set up legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=False,
        frameon=False, shadow=False, ncol=2, fontsize=20)

    # Write out to
    plt.savefig('robust.respy.png', bbox_inches='tight', format='png')

