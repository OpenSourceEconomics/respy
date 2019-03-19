import numpy as np

from respy.python.shared.shared_auxiliary import get_conditional_probabilities
from respy.python.evaluate.evaluate_auxiliary import (
    get_smoothed_probability,
    get_pdf_of_normal_distribution,
)
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.python.shared.shared_auxiliary import get_continuation_value


def pyth_contributions(
    state_space,
    data,
    periods_draws_prob,
    tau,
    num_agents_est,
    num_obs_agent,
    optim_paras,
):
    """Likelihood contribution of each individual in the sample.

    The code allows for a deterministic model (i.e. without randomness in the rewards).
    If that is the case and all agents have corresponding experiences, then one is
    returned. If a single agent violates the implications, then the zero is returned.

    Parameters
    ----------
    state_space : class
        Class of state space.
    data : pd.DataFrame
        DataFrame with the empirical dataset.
    periods_draws_prob : np.ndarray
        Array with shape (num_periods, num_draws_prob, num_choices) containing i.i.d.
        draws from standard normal distributions.
    tau : float
        Smoothing parameter for choice probabilities.
    num_agents_est : int
        Number of observations used for estimation.
    num_obs_agent : np.ndarray
        Array with shape (num_agents_est,) that contains the number of observed
        observations for each individual in the sample.
    optim_paras : dict
        Dictionary with quantities that were extracted from the parameter vector.

    Returns
    -------
    contribs : np.ndarray
        Array with shape (num_agents_est,) containing contributions of estimated agents.

    """
    num_obs_agent = num_obs_agent.astype(int)
    num_draws_prob = periods_draws_prob.shape[1]

    # Convert data to np.ndarray which is faster in every aspect. Separate wages from
    # other characteristics as they need to be integers.
    agents = data[
        [
            "Period",
            "Experience_A",
            "Experience_B",
            "Years_Schooling",
            "Lagged_Choice",
            "Choice",
        ]
    ].values.astype(int)
    wages = data["Wage"].values

    # Define a shortcut to the Cholesky factors of the shocks, because they are so
    # frequently used inside deeply nested loops.
    sc = optim_paras["shocks_cholesky"]

    # Initialize auxiliary objects.
    contribs = np.full(num_agents_est, np.nan)

    # Calculate the probability over agents and time.
    for j in range(num_agents_est):

        row_start = num_obs_agent[:j].sum()
        num_obs = num_obs_agent[j]
        edu_start = agents[row_start, 3]

        # Update type probabilities conditional on edu_start > 9.
        type_shares = get_conditional_probabilities(
            optim_paras["type_shares"], edu_start
        )

        # Container for the likelihood of the observed choice for each type. Likelihood
        # contribution for each type
        prob_type = np.ones(state_space.num_types)

        for type_ in range(state_space.num_types):

            prob_obs = np.zeros(num_obs)

            for p in range(num_obs):
                # Extract observable components of the state space as well as agent's
                # decision.
                period, exp_a, exp_b, edu, choice_lagged, choice = agents[
                    row_start + p
                ]
                wage_observed = wages[row_start + p]

                # Extract relevant deviates from standard normal distribution. The same
                # set of baseline draws are used for each agent and period. The copy is
                # needed as the object is otherwise changed in-place.
                draws_stan = periods_draws_prob[period].copy()

                # Get state index to access the systematic component of the agents
                # rewards. These feed into the simulation of choice probabilities.
                k = state_space.indexer[
                    period, exp_a, exp_b, edu, choice_lagged - 1, type_
                ]

                # If an agent is observed working, then the the labor market shocks are
                # observed and the conditional distribution is used to determine the
                # choice probabilities if the wage information is available as well.
                is_working = choice in [1, 2]
                has_wage = not np.isnan(wage_observed)

                if is_working and has_wage:
                    wage_systematic = state_space.rewards[k, -2:][choice - 1]

                    # Calculate the disturbance which are implied by the model and the
                    # observed wages.
                    dist = np.clip(
                        np.log(wage_observed), -HUGE_FLOAT, HUGE_FLOAT
                    ) - np.clip(
                        np.log(wage_systematic), -HUGE_FLOAT, HUGE_FLOAT
                    )

                    # Construct independent normal draws implied by the agents state
                    # experience. This is needed to maintain the correlation structure
                    # of the disturbances. Special care is needed in case of a
                    # deterministic model, as otherwise a zero division error occurs.
                    if choice == 1:
                        draws_stan[:, 0] = dist / sc[0, 0]
                        means = np.zeros(num_draws_prob)
                    elif choice == 2:
                        draws_stan[:, 1] = (
                            dist - sc[1, 0] * draws_stan[:, 0]
                        ) / sc[1, 1]
                        means = sc[1, 0] * draws_stan[:, 0]

                    sd = np.abs(sc[choice - 1, choice - 1])

                    prob_wages = get_pdf_of_normal_distribution(
                        dist, means, sd
                    )

                else:
                    prob_wages = np.ones(num_draws_prob)

                draws = draws_stan.dot(sc.T)

                draws[:, :2] = np.clip(np.exp(draws[:, :2]), 0.0, HUGE_FLOAT)

                total_values, _ = get_continuation_value(
                    state_space.rewards[k, -2:],
                    state_space.rewards[k, :4],
                    state_space.emaxs[k, :4],
                    draws,
                    optim_paras["delta"],
                )

                total_values = total_values.reshape(4, -1).T

                prob_choices = get_smoothed_probability(
                    total_values, choice - 1, tau
                )

                # Determine relative shares
                prob_obs[p] = prob_choices.dot(prob_wages) / num_draws_prob

            prob_type[type_] = np.prod(prob_obs)

        # Adjust and record likelihood contribution.
        contribs[j] = prob_type.dot(type_shares)

    return contribs
