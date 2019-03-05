from scipy.stats import norm
import numpy as np

from respy.python.shared.shared_auxiliary import get_conditional_probabilities
from respy.python.evaluate.evaluate_auxiliary import get_smoothed_probability
from respy.python.shared.shared_constants import SMALL_FLOAT
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.python.shared.shared_auxiliary import get_continuation_value


def pyth_contributions(
    state_space,
    data,
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

    The code allows for a deterministic model (i.e. without randomness in the rewards).
    If that is the case and all agents have corresponding experiences, then one is
    returned. If a single agent violates the implications, then the zero is returned.

    Parameters
    ----------
    state_space
    data : pd.DataFrame with the empirical dataset
    periods_draws_prob : np.array
        3d numpy array of dimension [nperiods, ndraws_prob, nchoices]. Contains iid
        draws from standard normal distributions.
    tau : float
        Smoothing parameter for choice probabilities.
    num_periods : int
        Number of periods of the model
    num_draws_prob : int
        Number of draws in the Monte Carlo integration of the choice probabilities.
    num_agents_est : int
        Number of observations used for estimation.
    num_obs_agent : np.array
        1d numpy array with length num_agents_est that contains the number of observed
        observations for each individual in the sample.
    num_types : int
        Number of types that govern unobserved heterogeneity.
    edu_spec : dict
    optim_paras : dict
        Dictionary with quantities that were extracted from the parameter vector.

    Returns
    -------
    ???

    """
    # Define a shortcut to the Cholesky factors of the shocks, because they are so
    # frequently used inside deeply nested loops
    sc = optim_paras["shocks_cholesky"]
    is_deterministic = np.count_nonzero(sc) == 0

    # Initialize auxiliary objects
    contribs = np.full(num_agents_est, -HUGE_FLOAT)
    prob_obs = np.full(num_periods, -HUGE_FLOAT)

    # Calculate the probability over agents and time.
    for j in range(num_agents_est):

        row_start = sum(num_obs_agent[:j])
        num_obs = num_obs_agent[j]
        edu_start = data.iloc[row_start]["Years_Schooling"].astype(int)

        # updated type probabilities, conditional on edu_start >= 9 or <= 9
        type_shares = get_conditional_probabilities(
            optim_paras["type_shares"], edu_start
        )

        # Container for the likelihood of the observed choice for each type. Likelihood
        # contribution for each type
        prob_type = np.ones(num_types)

        for type_ in range(num_types):

            # prob_obs has length p
            prob_obs[:] = 0.00
            for p in range(num_obs):

                agent = data.iloc[row_start + p]

                period = int(agent["Period"])
                # Extract observable components of state space as well as agent
                # decision.
                exp_a, exp_b, edu, choice_lagged, choice, wage_observed = agent[
                    [
                        "Experience_A",
                        "Experience_B",
                        "Years_Schooling",
                        "Lagged_Choice",
                        "Choice",
                        "Wage",
                    ]
                ]

                # Determine whether the agent's wage is known
                is_wage_missing = np.isnan(wage_observed)
                is_working = choice in [1, 2]

                # TODO: Type conversion for tests.
                # Create an index for the choice.
                idx = int(choice - 1)

                # Extract relevant deviates from standard normal distribution. The same
                # set of baseline draws are used for each agent and period. The copy is
                # needed as the object is otherwise changed inplace.
                draws_prob_raw = periods_draws_prob[period, :, :].copy()

                # Get state index to access the systematic component of the agents
                # rewards. These feed into the simulation of choice probabilities.
                state = state_space[
                    period, exp_a, exp_b, edu, choice_lagged - 1, type_
                ]

                # If an agent is observed working, then the the labor market shocks are
                # observed and the conditional distribution is used to determine the
                # choice probabilities if the wage information is available as well.
                if is_working and (not is_wage_missing):
                    wages_systematic = state[["wage_a", "wage_b"]].values

                    # Calculate the disturbance which are implied by the model and the
                    # observed wages.
                    dist = np.clip(
                        np.log(wage_observed), -HUGE_FLOAT, HUGE_FLOAT
                    ) - np.clip(
                        np.log(wages_systematic[idx]), -HUGE_FLOAT, HUGE_FLOAT
                    )

                    # If there is no random variation in rewards, then the observed
                    # wages need to be identical to their systematic components. The
                    # discrepancy between the observed wages and their systematic
                    # components might be nonzero (but small) due to the reading in of
                    # the dataset (FORTRAN only).
                    if is_deterministic and (dist > SMALL_FLOAT):
                        contribs[:] = 1
                        return contribs

                # Simulate the conditional distribution of alternative-specific value
                # functions and determine the choice probabilities.
                counts = np.full(4, 0)

                for s in range(num_draws_prob):

                    # Extract the standard normal deviates for the iteration.
                    draws_stan = draws_prob_raw[s, :]

                    # Construct independent normal draws implied by the agents state
                    # experience. This is needed to maintain the correlation structure
                    # of the disturbances. Special care is needed in case of a
                    # deterministic model, as otherwise a zero division error occurs.
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

                    # As deviates are aligned with the state experiences, create the
                    # conditional draws. Note, that the realization of the random
                    # component of wages align with their observed counterpart in the
                    # data.
                    draws_cond = sc.dot(draws_stan.T).T

                    # Extract deviates from (un-)conditional normal distributions and
                    # transform labor market shocks. This makes a copy.
                    draws = draws_cond[:]
                    draws[:2] = np.clip(np.exp(draws[:2]), 0.0, HUGE_FLOAT)

                    # Calculate total values. immediate rewards, including shock +
                    # expected future value!
                    total_values, _ = get_continuation_value(
                        state[["wage_a", "wage_b"]].values,
                        state[
                            [
                                "rewards_systematic_a",
                                "rewards_systematic_b",
                                "rewards_systematic_edu",
                                "rewards_systematic_home",
                            ]
                        ].values,
                        draws.reshape(1, -1),
                        state[
                            ["emaxs_a", "emaxs_b", "emaxs_edu", "emaxs_home"]
                        ].values,
                        optim_paras["delta"],
                    )
                    total_values = total_values.ravel()

                    # Record optimal choices
                    counts[np.argmax(total_values)] += 1

                    # Get the smoothed choice probability.
                    prob_choice = get_smoothed_probability(
                        total_values, idx, tau
                    )
                    prob_obs[p] += prob_choice * prob_wage

                # Determine relative shares
                prob_obs[p] = prob_obs[p] / num_draws_prob

                # If there is no random variation in rewards, then this implies that the
                # observed choice in the dataset is the only choice.
                if is_deterministic and (not (counts[idx] == num_draws_prob)):
                    contribs[:] = 1
                    return contribs

            prob_type[type_] = np.prod(prob_obs[:num_obs])

        # Adjust  and record likelihood contribution
        contribs[j] = np.sum(prob_type * type_shares)

    # If there is no random variation in rewards and no agent violated the implications
    # of observed wages and choices, then the evaluation return value of one.
    if is_deterministic:
        contribs[:] = np.exp(1.0)

    return contribs
