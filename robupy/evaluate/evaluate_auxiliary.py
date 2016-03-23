""" This module contains some auxiliary functions for the evaluation of the
criterion function.
"""
# standard library
from scipy.stats import norm

import numpy as np

# project library
from robupy.shared.auxiliary import get_total_value, check_dataset
from robupy.shared.constants import SMALL_FLOAT
from robupy.shared.constants import TINY_FLOAT
from robupy.shared.constants import HUGE_FLOAT


def evaluate_criterion_function(mapping_state_idx, periods_emax,
        periods_payoffs_systematic, states_all, shocks_cov, edu_max, delta,
        edu_start, num_periods, shocks_cholesky, num_agents, num_draws_prob,
        data_array, periods_draws_prob, is_deterministic):
    """ Evaluate criterion function. This code allows for a deterministic
    model, where there is no random variation in the rewards. If that is the
    case and all agents have corresponding experiences, then one is returned.
    If a single agent violates the implications, then the zero is returned.
    """
    # Initialize auxiliary objects
    likl, j = [], 0

    # Calculate the probability over agents and time.
    for i in range(num_agents):
        for period in range(num_periods):
            # Extract observable components of state space as well as agent
            # decision.
            exp_a, exp_b, edu, edu_lagged = data_array[j, 4:].astype(int)
            choice = data_array[j, 2].astype(int)
            is_working = choice in [1, 2]

            # Transform total years of education to additional years of
            # education and create an index from the choice.
            edu, idx = edu - edu_start, choice - 1

            # Get state indicator to obtain the systematic component of the
            # agents payoffs. These feed into the simulation of choice
            # probabilities.
            k = mapping_state_idx[period, exp_a, exp_b, edu, edu_lagged]
            payoffs_systematic = periods_payoffs_systematic[period, k, :]

            # Extract relevant deviates from standard normal distribution.
            draws_prob = periods_draws_prob[period, :, :].copy()

            # Prepare to calculate product of likelihood contributions.
            likl_contrib = 1.0

            # If an agent is observed working, then the the labor market shocks
            # are observed and the conditional distribution is used to determine
            # the choice probabilities.
            if is_working:
                # Calculate the disturbance, which follows a normal
                # distribution.
                dist = np.log(data_array[j, 3].astype(float)) - \
                        np.log(payoffs_systematic[idx])

                # If there is no random variation in payoffs, then the
                # observed wages need to be identical their systematic
                # components. The discrepancy between the observed wages and
                # their systematic components might be small due to the
                # reading in of the dataset (FORTRAN only).
                if is_deterministic and (dist > SMALL_FLOAT):
                    return 0.0

                # Construct independent normal draws implied by the observed
                # wages.
                if choice == 1:
                    draws_prob[:, idx] = dist / np.sqrt(shocks_cov[idx, idx])
                else:
                    draws_prob[:, idx] = (dist - shocks_cholesky[idx, 0] *
                        draws_prob[:, 0]) / shocks_cholesky[idx, idx]

                # Record contribution of wage observation.
                likl_contrib *= norm.pdf(dist, 0.0, np.sqrt(shocks_cov[idx, idx]))
            # Determine conditional deviates. These correspond to the
            # unconditional draws if the agent did not work in the labor market.
            conditional_draws = np.dot(shocks_cholesky, draws_prob.T).T

            # Simulate the conditional distribution of alternative-specific
            # value functions and determine the choice probabilities.
            counts = np.tile(0.0, 4)

            for s in range(num_draws_prob):
                # Extract deviates from (un-)conditional normal distributions.
                draws = conditional_draws[s, :]

                draws[0] = np.exp(draws[0])
                draws[1] = np.exp(draws[1])

                # Calculate total payoff.
                total_payoffs, _, _ = get_total_value(period, num_periods,
                    delta, payoffs_systematic, draws, edu_max, edu_start,
                    mapping_state_idx, periods_emax, k, states_all)

                # Record optimal choices
                counts[np.argmax(total_payoffs)] += 1.0

            # Determine relative shares
            choice_probabilities = counts / num_draws_prob

            # If there is no random variation in payoffs, then this implies a
            # unique optimal choice.
            if is_deterministic and (not (max(counts) == num_draws_prob)):
                return 0.0

            # Adjust  and record likelihood contribution
            likl_contrib *= choice_probabilities[idx]
            likl += [likl_contrib]

            j += 1

    # Scaling
    likl = -np.mean(np.log(np.clip(likl, TINY_FLOAT, HUGE_FLOAT)))

    # If there is no random variation in payoffs and no agent violated the
    # implications of observed wages and choices, then the evaluation return
    # a value of one.
    if is_deterministic:
        likl = 1.0

    # Finishing
    return likl


def check_evaluation(str_, *args):
    """ Check integrity of criterion function.
    """
    if str_ == 'out':

        # Distribute input parameters
        likl, = args

        # Check quality
        assert isinstance(likl, float)
        assert np.isfinite(likl)

    elif str_ == 'in':

        # Distribute input parameters
        data_frame, robupy_obj, is_deterministic = args

        # Check quality
        check_dataset(data_frame, robupy_obj)

    # Finishing
    return True