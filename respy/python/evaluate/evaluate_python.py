from scipy.stats import norm
import numpy as np

from respy.python.shared.shared_auxiliary import get_conditional_probabilities
from respy.python.evaluate.evaluate_auxiliary import get_smoothed_probability
from respy.python.shared.shared_auxiliary import back_out_systematic_wages
from respy.python.shared.shared_auxiliary import get_total_values
from respy.python.shared.shared_constants import SMALL_FLOAT
from respy.python.shared.shared_constants import HUGE_FLOAT


def pyth_contributions(
    periods_rewards_systematic,
    mapping_state_idx,
    periods_emax,
    states_all,
    data_array,
    periods_draws_prob,
    tau,
    num_periods,
    num_draws_prob,
    num_agents_est,
    num_obs_agent,
    num_types,
    edu_spec,
    optim_paras,
):
    """Likelihood contribution of each individual in the sample.

    The code allows for a deterministic model (i.e. without randomness in the
    rewards). If that is the case and all agents have corresponding
    experiences, then one is returned.
    If a single agent violates the implications, then the zero is returned.

    Args:
        periods_rewards_systematic:

        mapping_state_idx:

        periods_emax:

        states_all: 3d numpy array of dimension [nperiods, 100000, 5], where
            5 is the number of state variables and 100000 is an upper bound
            for the state space size per period.

        data_array: 2d numpy array with the empirical dataset

        periods_draws_prob: 3d numpy array of dimension [nperiods, ndraws_prob,
            nchoices]. Contains iid draws from standard normal distributions.

        tau: Smoothing parameter for choice probabilities.

        num_periods: Number of periods of the model

        num_draws_prob (int): Number of draws in the Monte Carlo integration
            of the choice probabilities.

        num_agents_est (int): Number of observations used for estimation.

        num_obs_agent: 1d numpy array with length num_agents_est that contains
            the number of observed observations for each individual in the sample.

        num_types (int): Number of types that govern unobserved heterogeneity.

        edu_spec:

        optim_paras (dict): Dictionary with quantities that were extracted
            from the parameter vector.
    """
    # define a shortcut to the cholesky factors of the shocks, because they
    # are so frequently used inside deeply nested loops
    sc = optim_paras["shocks_cholesky"]
    is_deterministic = np.count_nonzero(sc) == 0

    # Initialize auxiliary objects
    contribs = np.tile(-HUGE_FLOAT, num_agents_est)
    prob_obs = np.tile(-HUGE_FLOAT, num_periods)

    # Calculate the probability over agents and time.
    for j in range(num_agents_est):

        row_start = sum(num_obs_agent[:j])
        num_obs = num_obs_agent[j]
        edu_start = data_array[row_start + 0, 6].astype(int)

        # updated type probabilities, conditional on edu_start >= 9 or <= 9
        type_shares = get_conditional_probabilities(
            optim_paras["type_shares"], edu_start
        )

        # Container for the likelihood of the observed choice for each type.
        # likelihood contribution for each type
        prob_type = np.tile(1.0, num_types)

        for type_ in range(num_types):

            # prob_obs has length p
            prob_obs[:] = 0.00
            for p in range(num_obs):

                period = int(data_array[row_start + p, 1])
                # Extract observable components of state space as well as
                # agent decision.
                exp_a, exp_b, edu, choice_lagged = data_array[
                    row_start + p, 4:8
                ].astype(int)
                choice = data_array[row_start + p, 2].astype(int)
                wage_observed = data_array[row_start + p, 3]

                # Determine whether the agent's wage is known
                is_wage_missing = np.isnan(wage_observed)
                is_working = choice in [1, 2]

                # Create an index for the choice.
                idx = choice - 1

                # Extract relevant deviates from standard normal distribution.
                # The same set of baseline draws are used for each agent and
                # period.
                draws_prob_raw = periods_draws_prob[period, :, :].copy()

                # Get state index to access the systematic component of the
                # agents rewards. These feed into the simulation of choice
                # probabilities.
                k = mapping_state_idx[
                    period, exp_a, exp_b, edu, choice_lagged - 1, type_
                ]
                rewards_systematic = periods_rewards_systematic[period, k, :]

                # If an agent is observed working, then the the labor market
                # shocks are observed and the conditional distribution is used
                # to determine the choice probabilities if the wage information
                # is available as well.
                if is_working and (not is_wage_missing):
                    wages_systematic = back_out_systematic_wages(
                        rewards_systematic,
                        exp_a,
                        exp_b,
                        edu,
                        choice_lagged,
                        optim_paras,
                    )
                    # Calculate the disturbance which are implied by the model
                    # and the observed wages.
                    dist = np.clip(
                        np.log(wage_observed), -HUGE_FLOAT, HUGE_FLOAT
                    ) - np.clip(np.log(wages_systematic[idx]), -HUGE_FLOAT, HUGE_FLOAT)

                    # If there is no random variation in rewards, then the
                    # observed wages need to be identical to their systematic
                    # components. The discrepancy between the observed
                    # wages and their systematic components might be nonzero
                    # (but small) due to the reading in of the dataset
                    # (FORTRAN only).
                    if is_deterministic and (dist > SMALL_FLOAT):
                        contribs[:] = 1
                        return contribs

                # Simulate the conditional distribution of alternative-specific
                # value functions and determine the choice probabilities.
                counts = np.tile(0, 4)

                for s in range(num_draws_prob):

                    # Extract the standard normal deviates for the iteration.
                    draws_stan = draws_prob_raw[s, :]

                    # Construct independent normal draws implied by the agents
                    # state experience. This is needed to maintain the
                    # correlation structure of the disturbances. Special care
                    # is needed in case of a deterministic model, as otherwise
                    # a zero division error occurs.
                    if is_working and (not is_wage_missing):
                        if is_deterministic:
                            prob_wage = HUGE_FLOAT
                        else:
                            if choice == 1:
                                draws_stan[0] = dist / sc[idx, idx]
                                mean = 0.00
                                sd = abs(sc[idx, idx])
                            else:
                                draws_stan[idx] = (
                                    dist - sc[idx, 0] * draws_stan[0]
                                ) / sc[idx, idx]
                                mean = sc[idx, 0] * draws_stan[0]
                                sd = abs(sc[idx, idx])

                            prob_wage = norm.pdf(dist, mean, sd)
                    else:
                        prob_wage = 1.0

                    # As deviates are aligned with the state experiences,
                    # create the conditional draws. Note, that the realization
                    # of the random component of wages align with their
                    # observed counterpart in the data.
                    draws_cond = np.dot(sc, draws_stan.T).T

                    # Extract deviates from (un-)conditional normal
                    # distributions and transform labor market shocks.
                    # this makes a copy
                    draws = draws_cond[:]
                    draws[:2] = np.clip(np.exp(draws[:2]), 0.0, HUGE_FLOAT)

                    # Calculate total values.
                    # immediate rewards, including shock + expected future value!
                    total_values, _ = get_total_values(
                        period,
                        num_periods,
                        optim_paras,
                        rewards_systematic,
                        draws,
                        edu_spec,
                        mapping_state_idx,
                        periods_emax,
                        k,
                        states_all,
                    )

                    # Record optimal choices
                    counts[np.argmax(total_values)] += 1

                    # Get the smoothed choice probability.
                    prob_choice = get_smoothed_probability(total_values, idx, tau)
                    prob_obs[p] += prob_choice * prob_wage

                # Determine relative shares
                prob_obs[p] = prob_obs[p] / num_draws_prob

                # If there is no random variation in rewards, then this implies
                # that the observed choice in the dataset is the only choice.
                if is_deterministic and (not (counts[idx] == num_draws_prob)):
                    contribs[:] = 1
                    return contribs

            prob_type[type_] = np.prod(prob_obs[:num_obs])

        # Adjust  and record likelihood contribution
        contribs[j] = np.sum(prob_type * type_shares)

    # If there is no random variation in rewards and no agent violated the
    # implications of observed wages and choices, then the evaluation return
    # value of one.
    if is_deterministic:
        contribs[:] = np.exp(1.0)

    # Finishing
    return contribs
