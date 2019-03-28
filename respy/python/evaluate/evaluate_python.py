import numpy as np

from respy.python.shared.shared_auxiliary import get_conditional_probabilities
from respy.python.evaluate.evaluate_auxiliary import (
    simulate_probability_of_agents_observed_choice,
    create_draws_and_prob_wages,
)


def pyth_contributions(
    state_space, data, periods_draws_prob, tau, optim_paras
):
    """Calculate the likelihood contribution of each individual in the sample.

    The function calculates all likelihood contributions for all observations in the
    data which means all individual-period-type combinations. Then, likelihoods are
    accumulated within each individual and type over all periods. After that, the result
    is multiplied with the type-specific shares which yields the contribution to the
    likelihood for each individual.

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
    optim_paras : dict
        Dictionary with quantities that were extracted from the parameter vector.

    Returns
    -------
    contribs : np.ndarray
        Array with shape (num_agents,) containing contributions of estimated agents.

    """
    # Convert data to np.ndarray. Separate wages from other characteristics as they need
    # to be integers.
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
    wages_observed = data["Wage"].values

    # Get the number of observations for each individual and an array with indices of
    # each individual's first observation. After that, extract initial education levels
    # per agent which are important for type-specific probabilities.
    num_obs_per_agent = np.bincount(data.Identifier.values)
    idx_agents_first_observation = np.hstack((0, np.cumsum(num_obs_per_agent)[:-1]))
    agents_initial_education_levels = agents[idx_agents_first_observation, 3]

    # Update type-specific probabilities conditional on whether the initial level of
    # education is greater than nine.
    type_shares = get_conditional_probabilities(
        optim_paras["type_shares"], agents_initial_education_levels
    )

    # Extract observable components of the state space and agent's decision.
    periods, exp_as, exp_bs, edus, choices_lagged, choices = (
        agents[:, i] for i in range(6)
    )

    # Get indices of states in the state space corresponding to all observations for all
    # types. The indexer has the shape (num_obs, num_types).
    ks = state_space.indexer[
        periods, exp_as, exp_bs, edus, choices_lagged - 1, :
    ]

    # Reshape periods, choices and wages_observed so that they match the shape (num_obs,
    # num_types) of the indexer.
    periods = state_space.states[ks, 0]
    choices = choices.repeat(state_space.num_types).reshape(
        -1, state_space.num_types
    )
    wages_observed = wages_observed.repeat(state_space.num_types).reshape(
        -1, state_space.num_types
    )
    wages_systematic = state_space.rewards[ks, -2:]

    # Adjust the draws to simulate the expected maximum utility and calculate the
    # probability of observing the wage.
    draws, prob_wages = create_draws_and_prob_wages(
        wages_observed,
        wages_systematic,
        periods,
        periods_draws_prob,
        choices,
        optim_paras["shocks_cholesky"],
    )

    # Simulate the probability of observing the choice of the individual.
    prob_choices = simulate_probability_of_agents_observed_choice(
        state_space.rewards[ks, -2:],
        state_space.rewards[ks, :4],
        state_space.emaxs[ks, :4],
        draws,
        optim_paras["delta"],
        choices - 1,
        tau,
    )

    # Multiply the probability of the agent's choice with the probability of wage and
    # average over all draws to get the probability of the observation.
    prob_obs = (prob_choices * prob_wages).mean(axis=2)

    # Accumulate the likelihood of observations for each individual-type combination
    # over all periods.
    prob_type = np.multiply.reduceat(prob_obs, idx_agents_first_observation)

    # Multiply each individual-type contribution with its type-specific shares and sum
    # over types to get the likelihood contribution for each individual.
    contribs = (prob_type * type_shares).sum(axis=1)

    return contribs
