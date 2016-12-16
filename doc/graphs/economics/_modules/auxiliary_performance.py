""" Some auxiliary functions for the evaluation of alternative decision rules.
"""
import pickle as pkl
import numpy as np
import shlex

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from respy.python.shared.shared_constants import INADMISSIBILITY_PENALTY
from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import create_draws
from respy.python.shared.shared_auxiliary import get_emaxs

from auxiliary_economics import get_float_directories
from auxiliary_economics import float_to_string


def run(decision_levels):
    """ We now evaluate the performance of alternative decision rules under different scenarios.
    """
    # We extract some information that is constant across all different models of the world.
    base_obj = pkl.load(open('../grid/rslt/' + float_to_string(0.00) +
                             '/solution.respy.pkl', 'rb'))
    periods_emax, num_periods, num_agents_sim, seed_sim, delta, mapping_state_idx, model_paras, \
        edu_start, edu_max, states_all, periods_rewards_systematic = dist_class_attributes(base_obj,
            'periods_emax', 'num_periods', 'num_agents_sim', 'seed_sim', 'delta',
            'mapping_state_idx', 'model_paras', 'edu_start', 'edu_max', 'states_all',
            'periods_rewards_systematic')

    # Distribute model parameters
    shocks_cholesky = model_paras['shocks_cholesky']

    # We can now create the set of relevant disturbances. The required mean-shifts is implemented
    # within the loop.
    periods_draws_sims_transformed = np.tile(np.nan, (num_periods, num_agents_sim, 4))
    periods_draws_sims = create_draws(num_periods, num_agents_sim, seed_sim, False)
    for period in range(num_periods):
        periods_draws_sims_transformed[period, :, :] = transform_disturbances(
            periods_draws_sims[period, :, :], shocks_cholesky)

    # Now we are ready to simulate the performance of each decision rule.
    rslt = dict()

    for decision_level in decision_levels:

        rslt[decision_level] = []

        # First we read in the model of the world that informs the decision making.
        fname = '../grid/rslt/' + float_to_string(decision_level) + \
                '/solution.respy.pkl'
        periods_emax = pkl.load(open(fname, 'rb')).get_attr('periods_emax')

        # We now iterate over all available models of the world.
        for world_level in get_float_directories('../grid/rslt'):
            # If applicable, check for the results from the worst-case determination.
            is_ambiguity = (world_level > 0)
            if is_ambiguity:
                shift_dict = get_shifts(world_level)
            else:
                shift_dict = None

            # Simulate agent experiences
            count = 0

            # Initialize data
            lifetime_utilities = []

            for i in range(num_agents_sim):

                current_state = states_all[0, 0, :].copy()

                lifetime_utility = 0.0

                # Iterate over each period for the agent
                for period in range(num_periods):

                    # Distribute state space
                    exp_a, exp_b, edu, edu_lagged = current_state

                    k = mapping_state_idx[period, exp_a, exp_b, edu, edu_lagged]

                    # Select relevant subset
                    rewards_systematic = periods_rewards_systematic[period, k, :]
                    draws = periods_draws_sims_transformed[period, i, :].copy()

                    # If applicable, implementing the shift due to the ambiguity.

                    if is_ambiguity:
                        print(shift_dict[period].keys())

                        shifts = shift_dict[period][k]
                    else:
                        shifts = [0.00, 0.00]

                    # Get total value of admissible states
                    draws[:2] += shifts

                    total_values, rewards_ex_post = get_total_values(period, num_periods, delta,
                        rewards_systematic, draws, edu_max, edu_start, mapping_state_idx,
                        periods_emax, k, states_all)

                    # Determine optimal choice
                    max_idx = np.argmax(total_values)

                    lifetime_utility += (delta ** period) * rewards_ex_post[max_idx]

                    # Update work experiences and education
                    if max_idx == 0:
                        current_state[0] += 1
                    elif max_idx == 1:
                        current_state[1] += 1
                    elif max_idx == 2:
                        current_state[2] += 1

                    # Update lagged education
                    current_state[3] = 0

                    if max_idx == 2:
                        current_state[3] = 1

                    # Update row indicator
                    count += 1

                lifetime_utilities.append(lifetime_utility)

            rslt[decision_level] += [np.mean(lifetime_utilities)]

    pkl.dump(rslt, open('performance.respy.pkl', 'wb'))


def plot():
    """ Plot performance of alternative decision rules.
    """

    rslt = pkl.load(open('performance.respy.pkl', 'rb'))

    world_levels = get_float_directories('../grid')
    decision_levels = rslt.keys()

    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    # # Baseline
    labels = ['Optimal', 'Robust']
    colors = ['black', 'red']

    for i, level in enumerate(decision_levels):
        yvalues = rslt[level]
        xvalues = world_levels
        ax.plot(xvalues, yvalues, label=labels[i], linewidth=5, color=colors[i])

    # Both axes
    ax.tick_params(labelsize=18, direction='out', axis='both', top='off', right='off')

    # Remove first element on y-axis
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    #ax.set_xlim([1 + 15, MAX_PERIOD + 15]), ax.set_ylim([0, 0.60])

    # # labels
    ax.set_xlabel('Ambiguity', fontsize=16)
    ax.set_ylabel('Lifetime Utility', fontsize=16)

    # Set up legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=False, frameon=False,
        shadow=False, ncol=2, fontsize=20)

    # Write out to
    plt.savefig('performance.respy.png', bbox_inches='tight', format='png')


def get_shifts(level):
    """ We read in the results from the worst-case determination.
    """
    shift_dict = dict()
    for line in open('../grid/rslt/' + float_to_string(level) +
            '/data.respy.amb').readlines():
        # Split line
        list_ = shlex.split(line)

        # Skip lines that offer not information.
        if len(list_) < 3:
            continue

        is_start_info = (list_[0] == 'PERIOD')
        is_info = list_[1] in ['A', 'B']

        if is_start_info:
            period, state = int(list_[1]), int(list_[3])

            # Ensure that period is still in set of keys.
            if period not in shift_dict.keys():
                shift_dict[period] = dict()

            shift_dict[period][state] = []

        if is_info:
            shift_dict[period][state] += [float(list_[2])]

    return shift_dict


def get_total_values(period, num_periods, delta, rewards_systematic, draws, edu_max, edu_start,
        mapping_state_idx, periods_emax, k, states_all):
    """ Get total value of all possible states.
    """
    # Initialize containers
    rewards_ex_post = np.tile(np.nan, 4)

    # Calculate ex post rewards
    for j in [0, 1]:
        rewards_ex_post[j] = rewards_systematic[j] * draws[j]

    for j in [2, 3]:
        rewards_ex_post[j] = rewards_systematic[j] + draws[j]

    # Get future values
    if period != (num_periods - 1):
        emaxs, is_inadmissible = get_emaxs(edu_max, edu_start, mapping_state_idx, period,
            periods_emax, k, states_all)
    else:
        is_inadmissible = False
        emaxs = np.tile(0.0, 4)

    # Calculate total utilities
    total_values = rewards_ex_post + delta * emaxs

    # This is required to ensure that the agent does not choose any inadmissible states. If the
    # state is inadmissible emaxs takes value zero. This aligns the treatment of inadmissible
    # values with the original paper.
    if is_inadmissible:
        total_values[2] += INADMISSIBILITY_PENALTY

    # Finishing
    return total_values, rewards_ex_post
