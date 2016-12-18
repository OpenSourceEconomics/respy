import matplotlib.pylab as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import pickle as pkl
import numpy as np

import shlex
import os

import respy

from auxiliary_economics import move_subdirectory
from auxiliary_economics import INIT_FILE

EDU, EXP_A, EXP_B = 10.00, 5, 5


def run():
    """ Create the input to plot some baseline information about the
    specification.
    """
    move_subdirectory()

    respy_obj = respy.RespyCls(INIT_FILE)
    respy_obj.unlock()
    respy_obj.set_attr('num_procs', 3)
    respy_obj.lock()

    respy.simulate(respy_obj)

    choice_probabilities = get_choice_probabilities('data.respy.info')
    pkl.dump(choice_probabilities, open('probabilities.respy.pkl', 'wb'))

    os.chdir('../')


def plot():
    """ Plot some baseline information.
    """

    os.chdir('rslt')

    plot_choice_patterns()
    plot_return_education()
    plot_return_experience()

    os.chdir('../')


def get_coeffs():
    """ Extract coefficients from the initialization file.
    """
    move_subdirectory()
    respy_obj = respy.RespyCls(INIT_FILE)
    model_paras = respy_obj.get_attr('model_paras')
    coeffs = dict()
    for label in ['a', 'b']:
        coeffs[label] = model_paras['coeffs_' + label]
    return coeffs


def wage_function(edu, exp_A, exp_B, coeffs):
    """ This function calculates the expected wage based on an agent's
    covariates for a given parametrization.
    """

    wage = coeffs[0]
    wage += coeffs[1] * edu
    wage += coeffs[2] * exp_A
    wage += coeffs[3] * exp_A ** 2
    wage += coeffs[4] * exp_B
    wage += coeffs[5] * exp_B ** 2

    wage = np.exp(wage)

    return wage


def return_to_experience(exp_A, exp_B, coeffs, which):
    """ Wrapper to evaluate the wage function for varying levels of experience.
    """
    # Get wage
    wage = wage_function(EDU, exp_A, exp_B, coeffs[which])

    # Finishing
    return wage


def return_to_education(edu, coeffs, which):
    """ Wrapper to evaluate the wage function for varying levels of education
    """
    wage = wage_function(edu, EXP_A, EXP_B, coeffs[which])

    return wage


def get_choice_probabilities(fname):
    """ Get the choice probabilities.
    """
    stats = np.tile(np.nan, (0, 4))

    with open(fname) as in_file:
        for line in in_file.readlines():
            list_ = shlex.split(line)
            if not list_:
                continue

            if list_[0] == 'Outcomes':
                break

            try:
                int(list_[0])
            except ValueError:
                continue

            stats = np.vstack((stats, [float(x) for x in list_[1:]]))

    # Finishing
    return stats


def plot_return_experience():
    """ Function to produce plot for the return to experience.
    """

    def _beautify_subplot(subplot, zlim):
        subplot.view_init(azim=180 + 40)

        subplot.set_ylabel('Experience A')
        subplot.set_xlabel('Experience B')
        subplot.set_zlabel('Wages')

        subplot.zaxis.set_rotate_label(False)
        subplot.set_zlabel(r'Wages (in \$1,000)', rotation=90)

        subplot.zaxis.get_major_ticks()[0].set_visible(False)

        # Background Color (higher numbers are lighter)
        subplot.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
        subplot.w_yaxis.set_pane_color((0.6, 0.6, 0.6, 1.0))
        subplot.w_zaxis.set_pane_color((0.68, 0.68, 0.68, 1.0))

        ax.set_zlim(zlim)

    coeffs = get_coeffs()

    z = dict()
    for which in ['a', 'b']:
        x, y = np.meshgrid(range(20), range(20))
        z[which] = np.tile(np.nan, (20, 20))
        for i in range(20):
            for j in range(20):
                args = [i, j, coeffs, which]
                z[which][i, j] = return_to_experience(*args)

    # Scaling
    z['a'] /= 1000
    z['b'] /= 1000

    zlim = [0, 30]

    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(x, y, z['a'], rstride=1, cstride=1, cmap=cm.jet,
                    linewidth=0, antialiased=False, alpha=0.8)
    _beautify_subplot(ax, zlim)

    ax = fig.add_subplot(122, projection='3d')
    ax.plot_surface(x, y, z['b'], rstride=1, cstride=1, cmap=cm.jet,
                    linewidth=0, antialiased=False, alpha=0.8)
    _beautify_subplot(ax, zlim)

    # Write out to
    plt.savefig('returns_experience.png', bbox_inches='tight', format='png')


def plot_return_education():
    """ Function to produce plot for the return to education.
    """

    coeffs = get_coeffs()

    # Determine wages for varying years of education in each occupation.
    xvals, yvals = range(10, 21), dict()
    for which in ['a', 'b']:
        yvals[which] = []
        for edu in xvals:
            yvals[which] += [return_to_education(edu, coeffs, which)]

    # Initialize plot
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    # Scaling
    for occu in ['a', 'b']:
        for i, _ in enumerate(xvals):
            yvals[occu][i] /= 1000

    # Draw lines
    ax.plot(xvals, yvals['a'], '-k', label='Occupation A', linewidth=5,
            color='red', alpha=0.8)
    ax.plot(xvals, yvals['b'], '-k', label='Occupation B', linewidth=5,
            color='orange', alpha=0.8)

    # Both axes
    ax.tick_params(labelsize=16, direction='out', axis='both', top='off',
        right='off')

    # x-axis
    ax.set_xticklabels(ax.get_xticks().astype(int))
    ax.set_xlabel('Years of Schooling', fontsize=16)

    # y-axis
    yticks = ['{:,.0f}'.format(y) for y in ax.get_yticks().astype(int)]
    ax.set_yticklabels(yticks, fontsize=16)

    ax.set_ylabel(r'Wages (in \$1,000)', fontsize=16)
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    # Set up legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
        fancybox=False, frameon=False, shadow=False, ncol=2, fontsize=20)

    # Write out to
    plt.savefig('returns_schooling.png', bbox_inches='tight', format='png')


def plot_choice_patterns():
    """ Function to produce plot for choice patterns.
    """
    choice_probabilities = pkl.load(open('probabilities.respy.pkl', 'rb'))
    deciles = range(40)
    colors = ['blue', 'yellow', 'orange', 'red']
    width = 0.9

    # Plotting
    bottom = [0] * 40

    # Initialize plot
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)
    labels = ['Home', 'School', 'Occupation A', 'Occupation B']
    for j, i in enumerate([3, 2, 0, 1]):
        heights = choice_probabilities[:, i]
        plt.bar(deciles, heights, width, bottom=bottom, color=colors[j],
                alpha=0.70)
        bottom = [heights[i] + bottom[i] for i in range(40)]

    # Both Axes
    ax.tick_params(labelsize=16, direction='out', axis='both', top='off',
        right='off')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # X axis
    ax.set_xlabel('Period', fontsize=16)
    ax.set_xlim([0, 40])

    # Y axis
    ax.set_ylabel('Share of Population', fontsize=16)
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    # Legend
    plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.10),
               fancybox=False, frameon=False, shadow=False, ncol=4,
               fontsize=20)

    # Write out to
    plt.savefig('choices.png', bbox_inches='tight',
                format='png')
