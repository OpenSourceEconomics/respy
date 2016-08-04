import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt


def plot_dimension_state_space(num_states):
    """ Plot the dimension of the state space
    """

    # Initialize plot
    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    ax.plot(range(1, 41), num_states, '-k', color='red',
                        linewidth=5, alpha=0.8)

    # Both axes
    ax.tick_params(axis='both', right='off', top='off')

    # x-axis
    ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=18)
    ax.set_xlabel('Period', fontsize=16)
    ax.set_xlim([1, 41])

    # y-axis
    yticks = ['{:,.0f}'.format(y) for y in ax.get_yticks().astype(int)]
    ax.set_yticklabels(yticks, fontsize=16)
    ax.set_ylabel('Number of Nodes', fontsize=16)
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    # Write out to
    plt.savefig('state_space.png', bbox_inches='tight', format='png')

