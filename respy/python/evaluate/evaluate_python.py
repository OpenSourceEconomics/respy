from scipy.stats import norm
import numpy as np

from respy.python.evaluate.evaluate_auxiliary import get_smoothed_probability
from respy.python.shared.shared_auxiliary import get_total_values
from respy.python.shared.shared_constants import SMALL_FLOAT
from respy.python.shared.shared_constants import HUGE_FLOAT


def pyth_contributions(periods_rewards_systematic, mapping_state_idx,
        periods_emax, states_all, data_array, periods_draws_prob, tau,
        edu_start, edu_max, num_periods, num_draws_prob, optim_paras):
    """ Evaluate criterion function. This code allows for a deterministic
    model, where there is no random variation in the rewards. If that is the
    case and all agents have corresponding experiences, then one is returned.
    If a single agent violates the implications, then the zero is returned.
    """
    # Construct auxiliary object
    shocks_cov = np.matmul(optim_paras['shocks_cholesky'],
        optim_paras['shocks_cholesky'].T)
    is_deterministic = (np.count_nonzero(optim_paras['shocks_cholesky']) == 0)
    num_obs = data_array.shape[0]

    # Initialize auxiliary objects
    contribs = np.tile(-HUGE_FLOAT, num_obs)

    # Calculate the probability over agents and time.
    for j in range(num_obs):
        period = int(data_array[j, 1])
        # Extract observable components of state space as well as agent
        # decision.
        exp_a, exp_b, edu, edu_lagged = data_array[j, 4:].astype(int)
        choice = data_array[j, 2].astype(int)
        wage = data_array[j, 3]

        # We now determine whether we also have information about the agent's
        # wage.
        is_wage_missing = np.isnan(wage)
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
        # the choice probabilities. At least if the wage information is
        # available as well.
        if is_working and (not is_wage_missing):
            # Calculate the disturbance which are implied by the model
            # and the observed wages.
            dist = np.clip(np.log(wage), -HUGE_FLOAT, HUGE_FLOAT) - \
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
            if is_working and (not is_wage_missing):
                if is_deterministic:
                    prob_wage = HUGE_FLOAT
                else:
                    if choice == 1:
                        draws_stan[0] = dist / optim_paras['shocks_cholesky'][
                            idx, idx]
                    else:
                        draws_stan[1] = (dist - optim_paras['shocks_cholesky'][idx, 0] *
                                         draws_stan[0]) / optim_paras['shocks_cholesky'][idx, idx]

                    prob_wage = norm.pdf(draws_stan[idx], 0.0, 1.0) / \
                                np.sqrt(shocks_cov[idx, idx])

            else:
                prob_wage = 1.0

            # As deviates are aligned with the state experiences, create
            # the conditional draws. Note, that the realization of the
            # random component of wages align withe their observed
            # counterpart in the data.
            draws_cond = np.dot(optim_paras['shocks_cholesky'], draws_stan.T).T

            # Extract deviates from (un-)conditional normal distributions
            # and transform labor market shocks.
            draws = draws_cond[:]
            draws[:2] = np.clip(np.exp(draws[:2]), 0.0, HUGE_FLOAT)

            # Calculate total values.
            total_values = get_total_values(period, num_periods, optim_paras,
                rewards_systematic, draws, edu_max, edu_start,
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
