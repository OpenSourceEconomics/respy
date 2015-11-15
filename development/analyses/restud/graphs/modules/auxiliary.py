""" This module contains auxiliary functions to plot some information on the 
RESTUD economy.
"""

# standard library
import matplotlib.pylab as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter
from matplotlib import cm

# Evaluation points
EDU, EXP_A, EXP_B = 15, 5, 5

""" Auxiliary function
"""


def wage_function(edu, exp_A, exp_B, coeffs):
    """ This function calculates the expected wage based on an agent's
    covariates for a given parameterization.
    """

    # Intercept
    wage = coeffs['int']

    # Schooling
    wage += coeffs['coeff'][0] * edu

    # Experience A
    wage += coeffs['coeff'][1] * exp_A
    wage += coeffs['coeff'][2] * exp_A ** 2

    # Experience B
    wage += coeffs['coeff'][3] * exp_B
    wage += coeffs['coeff'][4] * exp_B ** 2

    # Transformation
    wage = np.exp(wage)

    # Finishing
    return wage


def return_to_experience(exp_A, exp_B, coeffs, which):
    """ Wrapper to evaluate the wage function for varying levels of experience.
    """
    # Get wage
    wage = wage_function(EDU, exp_A, exp_B, coeffs[which])

    # Finishing
    return wage


# Auxiliary function
def return_to_education(edu, coeffs, which):
    """ Wrapper to evaluate the wage function for varying levels of education
    """
    # Get wage
    wage = wage_function(edu, EXP_A, EXP_B, coeffs[which])

    # Finishing
    return wage

""" Plotting functions
"""


def plot_dimension_state_space(num_states):
    """ Plot the dimension of the state space
    """

    # Initialize plot
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    ax.plot(range(1, 41), num_states, '-k', color='red',
                        linewidth=5)

    # Both axes
    ax.tick_params(axis='both', right='off', top='off')

    # x-axis
    ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=18)
    ax.set_xlabel('Periods', fontsize=16)
    ax.set_xlim([1, 41])

    # y-axis
    yticks = ['{:,.0f}'.format(y) for y in ax.get_yticks().astype(int)]
    ax.set_yticklabels(yticks, fontsize=16)
    ax.set_ylabel('Number of States', fontsize=16)
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    # Write out to
    plt.savefig('rslts/state_space.png', bbox_inches='tight',
                format='png')


def plot_return_experience(x, y, z, which, spec):
    """ Function to produce plot for the return to experience.
    """

    # Scaling
    z = z / 1000

    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.view_init(azim=180+40)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)

    # Axis labels.
    ax.set_ylabel('Experience A')
    ax.set_xlabel('Experience B')

    ax.zaxis.set_rotate_label(False)

    # Z axis ticks
    ax.set_zlim([ax.get_zlim()[0], ax.get_zlim()[1]])
    if spec == 'One':
        ax.set_zlim([15, 55])
    elif spec == 'Two':
        ax.set_zlim([10, 50])
    elif spec == 'Three':
        ax.set_zlim([10, 700])

    ax.zaxis.get_major_ticks()[0].set_visible(False)

    # Background Color (higher numbers are lighter)
    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_yaxis.set_pane_color((0.6, 0.6, 0.6, 1.0))
    ax.w_zaxis.set_pane_color((0.68, 0.68, 0.68, 1.0))

    # Write out to
    plt.savefig('rslts/data_' + spec.lower() + '/returns_experience_' +
                which.lower() + '.png', bbox_inches='tight', format='png')


def plot_return_education(xvals, yvals, spec):
    """ Function to produce plot for the return to education.
    """

    labels = ['Occupation A', 'Occupation B']

    # Initialize plot
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    # Scaling
    for occu in ['A', 'B']:
        for i in range(len(xvals)):
            yvals[occu][i] = yvals[occu][i] / 1000

    # Draw lines
    ax.plot(xvals, yvals['A'], '-k', label='Occupation A', linewidth=5,
            color='red')
    ax.plot(xvals, yvals['B'], '-k', label='Occupation B', linewidth=5,
            color='orange')

    # Both axes
    ax.tick_params(labelsize=16, direction='out', axis='both', top='off',
        right='off')

    # x-axis
    ax.set_xticklabels(ax.get_xticks().astype(int))
    ax.set_xlabel('Years of Schooling', fontsize=16)

    # y-axis
    yticks = ['{:,.0f}'.format(y) for y in ax.get_yticks().astype(int)]
    ax.set_yticklabels(yticks, fontsize=16)

    ax.set_ylabel('Wages', fontsize=16)
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    # Set up legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
        fancybox=False, frameon=False, shadow=False, ncol=2, fontsize=20)

    # Write out to
    plt.savefig('rslts/data_' + spec.lower() + '/returns_schooling.png',
                bbox_inches='tight', format='png')


def plot_choice_patterns(choice_probabilities, spec):
    """ Function to produce plot for choice patterns.
    """
    labels = ['Home', 'School', 'Occupation A', 'Occupation B']

    rows, cols = 4, 40

    deciles = range(40)

    colors = ['blue', 'yellow', 'orange', 'red']

    width = 0.9

    # Plotting
    bottom = [0]*40

    # Initialize plot
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    for i, row in enumerate(labels):

        heights = choice_probabilities[row][:]
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

    # Legend
    plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.10),
        fancybox=False, frameon=False, shadow=False, ncol=4, fontsize=20)

    # Write out to
    plt.savefig('rslts/data_' + spec.lower() + '/choice_patterns.png',
                bbox_inches='tight', format='png')

