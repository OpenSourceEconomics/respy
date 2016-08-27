from scipy.stats import norm
import numpy as np

from respy.python.solve.solve_auxiliary import pyth_calculate_rewards_systematic
from respy.python.evaluate.evaluate_auxiliary import get_smoothed_probability
from respy.python.solve.solve_auxiliary import pyth_backward_induction
from respy.python.shared.shared_auxiliary import get_total_values
from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.shared.shared_constants import SMALL_FLOAT
from respy.python.shared.shared_constants import HUGE_FLOAT


def pyth_contributions(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cholesky, is_interpolated, num_draws_emax, num_periods,
        num_points_interp, is_myopic, edu_start, is_debug, edu_max, delta,
        data_array, num_agents_est, num_draws_prob, tau, periods_draws_emax,
        periods_draws_prob, states_all, states_number_period,
        mapping_state_idx, max_states_period, is_ambiguity, level):
    """ Evaluate criterion function. This code allows for a deterministic
    model, where there is no random variation in the rewards. If that is the
    case and all agents have corresponding experiences, then one is returned.
    If a single agent violates the implications, then the zero is returned.
    """
    # Construct auxiliary object
    shocks_cov = np.matmul(shocks_cholesky, shocks_cholesky.T)
    is_deterministic = (np.count_nonzero(shocks_cholesky) == 0)

    # Calculate all systematic rewards
    periods_rewards_systematic = pyth_calculate_rewards_systematic(num_periods,
        states_number_period, states_all, edu_start, coeffs_a, coeffs_b,
        coeffs_edu, coeffs_home, max_states_period)

    # Initialize containers, which contain a lot of missing values as we
    # capture the tree structure in arrays of fixed dimension.
    i, j = num_periods, max_states_period
    periods_emax = np.tile(MISSING_FLOAT, (i, j))

    if is_myopic:
        # All other objects remain set to MISSING_FLOAT. This align the
        # treatment for the two special cases: (1) is_myopic and (2)
        # is_interpolated.
        for period, num_states in enumerate(states_number_period):
            periods_emax[period, :num_states] = 0.0

    else:
        periods_emax = pyth_backward_induction(num_periods, max_states_period,
            periods_draws_emax, num_draws_emax, states_number_period,
            periods_rewards_systematic, edu_max, edu_start,
            mapping_state_idx, states_all, delta, is_debug, is_interpolated,
            num_points_interp, shocks_cholesky, is_ambiguity, level)

    # Initialize auxiliary objects
    contribs = np.tile(-HUGE_FLOAT, (num_agents_est * num_periods))
    j = 0

    # Calculate the probability over agents and time.
    for _ in range(num_agents_est):
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
            # agents rewards. These feed into the simulation of choice
            # probabilities.
            k = mapping_state_idx[period, exp_a, exp_b, edu, edu_lagged]
            rewards_systematic = periods_rewards_systematic[period, k, :]

            # Extract relevant deviates from standard normal distribution.
            # The same set of baseline draws are used for each agent and period.
            draws_prob_raw = periods_draws_prob[period, :, :].copy()

            # If an agent is observed working, then the the labor market shocks
            # are observed and the conditional distribution is used to determine
            # the choice probabilities.
            if is_working:
                # Calculate the disturbance which are implied by the model
                # and the observed wages.
                dist = np.clip(np.log(data_array[j, 3]), -HUGE_FLOAT, HUGE_FLOAT) - \
                       np.clip(np.log(rewards_systematic[idx]), -HUGE_FLOAT,
                           HUGE_FLOAT)

                # If there is no random variation in rewards, then the
                # observed wages need to be identical their systematic
                # components. The discrepancy between the observed wages and
                # their systematic components might be small due to the
                # reading in of the dataset (FORTRAN only).
                if is_deterministic and (dist > SMALL_FLOAT):
                    contribs[:] = 1
                    return contribs

            # Simulate the conditional distribution of alternative-specific
            # value functions and determine the choice probabilities.
            counts, prob_obs = np.tile(0, 4), 0.0

            for s in range(num_draws_prob):

                # Extract the standard normal deviates sample for the iteration.
                draws_stan = draws_prob_raw[s, :]

                # Construct independent normal draws implied by the agents
                # state experience. This is need to maintain the correlation
                # structure of the disturbances.  Special care is needed in case
                # of a deterministic model, as otherwise a zero division error
                # occurs.
                if is_working:
                    if is_deterministic:
                        prob_wage = HUGE_FLOAT
                    else:
                        if choice == 1:
                            draws_stan[0] = dist / shocks_cholesky[idx, idx]
                        else:
                            draws_stan[1] = (dist - shocks_cholesky[idx, 0] *
                                draws_stan[0]) / shocks_cholesky[idx, idx]

                        prob_wage = norm.pdf(draws_stan[idx], 0.0, 1.0) / \
                            np.sqrt(shocks_cov[idx, idx])

                else:
                    prob_wage = 1.0

                # As deviates are aligned with the state experiences, create
                # the conditional draws. Note, that the realization of the
                # random component of wages align withe their observed
                # counterpart in the data.
                draws_cond = np.dot(shocks_cholesky, draws_stan.T).T

                # Extract deviates from (un-)conditional normal distributions
                # and transform labor market shocks.
                draws = draws_cond[:]
                draws[:2] = np.clip(np.exp(draws[:2]), 0.0, HUGE_FLOAT)

                # Calculate total values.
                total_values = get_total_values(period, num_periods,
                    delta, rewards_systematic, draws, edu_max, edu_start,
                    mapping_state_idx, periods_emax, k, states_all)

                # Record optimal choices
                counts[np.argmax(total_values)] += 1

                # Get the smoothed choice probability.
                prob_choice = get_smoothed_probability(total_values, idx, tau)
                prob_obs += prob_choice * prob_wage

            # Determine relative shares
            prob_obs = prob_obs / num_draws_prob

            # If there is no random variation in rewards, then this implies
            # that the observed choice in the dataset is the only choice.
            if is_deterministic and (not (counts[idx] == num_draws_prob)):
                contribs[:] = 1
                return contribs

            # Adjust  and record likelihood contribution
            contribs[j] = prob_obs

            j += 1

    # If there is no random variation in rewards and no agent violated the
    # implications of observed wages and choices, then the evaluation return
    # a value of one.
    if is_deterministic:
        contribs[:] = np.exp(1.0)

    # Finishing
    return contribs
