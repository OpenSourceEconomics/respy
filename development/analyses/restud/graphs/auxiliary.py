""" This module contains auxiliary functions to plot some information on the 
RESTUD economy.
"""

# standard library
import matplotlib.pylab as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter
from matplotlib import cm

#
EDU, EXP_A, EXP_B = 15, 15, 15

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
    ax = plt.subplot()

    ax.plot(range(40), num_states, '-k', )

    # Both axes
    ax.tick_params(axis='both', right='off', top='off')

    # x-axis
    ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=18)
    ax.set_xlabel('Periods', fontsize=24)

    # y-axis
    yticks = ['{:,.0f}'.format(y) for y in ax.get_yticks().astype(int)]
    ax.set_yticklabels(yticks, fontsize=18)
    ax.set_ylabel('Number of States', fontsize=18)

    # Write out to
    plt.savefig('rslts/state_space.pdf', bbox_inches='tight',
                format="pdf")


def plot_return_experience(x, y, z, which, spec):
    """ Function to produce plot for the return to experience.
    """

    fig = plt.figure()

    ax = fig.gca(projection = '3d')
    ax.view_init(azim = 180+40)
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet,
                linewidth=0, antialiased=False)

    # Axis labels.
    ax.set_ylabel('Experience A')
    ax.set_xlabel('Experience B')

    ax.zaxis.set_rotate_label(False)
    #ax.set_zlabel(zlabel, rotation=90)

    # Z axis ticks
    pad = 0.07*(ax.get_zlim()[1] - ax.get_zlim()[0])
    ax.set_zlim([ax.get_zlim()[0], ax.get_zlim()[1]])
    #if zlim is not None:
    #    ax.set_zlim(zlim)
    ax.zaxis.set_major_formatter(
            FuncFormatter('{:>4}'.format))

    # Background Color (higher numbers are lighter)
    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_yaxis.set_pane_color((0.6, 0.6, 0.6, 1.0))
    ax.w_zaxis.set_pane_color((0.68, 0.68, 0.68, 1.0))

    # Write out to
    plt.savefig('rslts/data_' + spec.lower() + '/returns_experience_' +
                which.lower() +'.pdf', bbox_inches='tight', format="pdf")


def plot_return_education(xvals, yvals, spec):
    """ Function to produce plot for the return to education.
    """
    # Initialize plot
    ax = plt.subplot()

    # Draw lines
    ax.plot(xvals, yvals['A'], '-k', label='Occupation A')
    ax.plot(xvals, yvals['B'], '--k', label='Occupation B')

    # Both axes
    ax.tick_params(axis='both', right='off', top='off')

    # x-axis
    ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=18)
    ax.set_xlabel('Years of Education', fontsize=18)

    # y-axis
    yticks = ['{:,.0f}'.format(y) for y in ax.get_yticks().astype(int)]
    ax.set_yticklabels(yticks, fontsize=18)
    ax.set_ylabel('Rewards', fontsize=18)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, ncol=2, frameon=False, fontsize=16)

    # Write out to
    plt.savefig('rslts/data_' + spec.lower() + '/returns_education.pdf',
                bbox_inches='tight', format="pdf")


def plot_choice_patterns(choice_probabilities, spec):
    """ Function to produce plot for choice patterns.
    """
    labels = ['Occupation A', 'Occupation B', 'Education', 'Home']

    rows, cols = 4, 40

    deciles = range(40)

    colors = [(0, k, 1) for k in np.linspace(0, 1, int(rows/3.))] + \
                [(0, 1, 1 - k) for k in np.linspace(0, 1, int(rows/3.) + 1)[1:]] + \
                [(0, 1, 1 - k) for k in np.linspace(0, 1, int(rows/3.) + 1)[1:]] + \
        [(k, 1 - k, 0) for k in np.linspace(0, 1, int(rows/3.) + 1)[1:]]

    width = 0.9

    # Plotting

    bottom = [0]*40

    ax = plt.subplot()

    for row in range(rows):

        heights = choice_probabilities[row][:]
        plt.bar(deciles, heights, width, bottom=bottom, color=colors[row])
        bottom = [heights[i] + bottom[i] for i in range(40)]

    # Both Axes
    ax.tick_params(labelsize=18, direction='out', axis='both', \
                        top='off', right='off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # X axis
    xticks = [d + width/2. for d in deciles]
    xticklabels = [str(d) for d in deciles]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=18)
    ax.set_xlabel('Decile', fontsize=18)
    ax.set_xlim([1, 40])

    # Y axis
    ax.set_ylabel('Share', fontsize=18)

    # Legend
    plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), \
                   fancybox=True, ncol=rows, fontsize=10)

    # Write out to
    plt.savefig('rslts/data_' + spec.lower() + '/choice_patterns.pdf',
                bbox_inches='tight', format="pdf")
