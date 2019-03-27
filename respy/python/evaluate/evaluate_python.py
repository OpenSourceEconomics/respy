import numpy as np

from respy.python.shared.shared_auxiliary import get_conditional_probabilities
from respy.python.evaluate.evaluate_auxiliary import (
    get_smoothed_probability,
    create_draws_for_monte_carlo_simulation,
    adjust_draws_and_create_prob_wages,
)
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.python.shared.shared_auxiliary import get_continuation_value


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
    # Convert data to np.ndarray which is faster. Separate wages from other
    # characteristics as they need to be integers.
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
    wages_observed = data["Wage"].values.reshape(-1, 1)

    # Get useful auxiliary objects.
    num_obs_per_agent = np.bincount(data.Identifier.values)
    num_obs = data.shape[0]
    sc = optim_paras["shocks_cholesky"]

    # Extend systematic wages with zeros so that indexing with choice three and four
    # does not fail.
    wages_systematic_ext = np.hstack(
        (state_space.rewards[:, -2:], np.zeros((state_space.num_states, 2)))
    )

    # Calculate the probability for all numbers of observations. The format of array is
    # (num_obs, num_types, num_draws, num_choices) reduced to the minimum of dimensions.
    # E.g. choices are determined for every agent thus the shape is (num_obs,),
    # prob_wages are calculated for each draw thus the shape is (num_obs, num_types,
    # num_draws).
    rows_start = np.hstack((0, np.cumsum(num_obs_per_agent)[:-1]))
    edu_starts = agents[rows_start, 3]

    # Update type probabilities conditional on edu_start > 9.
    type_shares = get_conditional_probabilities(
        optim_paras["type_shares"], edu_starts
    )

    # Extract observable components of the state space as well as agent's
    # decision.
    periods, exp_as, exp_bs, edus, choices_lagged, choices = (
        agents[:, i] for i in range(6)
    )

    # Get state index to access the systematic component of the agents
    # rewards. These feed into the simulation of choice probabilities.
    ks = state_space.indexer[
        periods, exp_as, exp_bs, edus, choices_lagged - 1, :
    ]

    wages_systematic = wages_systematic_ext[ks, choices.reshape(-1, 1) - 1]

    # Calculate the disturbance which are implied by the model and the observed wages.
    dist = np.clip(np.log(wages_observed), -HUGE_FLOAT, HUGE_FLOAT) - np.clip(
        np.log(wages_systematic), -HUGE_FLOAT, HUGE_FLOAT
    )
    dist = dist.reshape(num_obs, state_space.num_types, 1)

    # Get periods based on the shape (num_obs, num_types) of the indexer.
    periods = state_space.states[ks, 0]

    c_ = choices.repeat(state_space.num_types).reshape(
        -1, state_space.num_types
    )

    draws_stan, prob_wages = adjust_draws_and_create_prob_wages(
        periods,
        periods_draws_prob,
        c_,
        dist.reshape(-1, state_space.num_types),
        sc,
    )

    draws = create_draws_for_monte_carlo_simulation(draws_stan, sc.T)

    total_values = get_continuation_value(
        state_space.rewards[ks, -2:],
        state_space.rewards[ks, :4],
        state_space.emaxs[ks, :4],
        draws,
        optim_paras["delta"],
    )

    total_values = total_values.transpose(0, 1, 3, 2)

    prob_choices = get_smoothed_probability(total_values, choices - 1, tau)

    # Determine relative shares
    prob_obs = (prob_choices * prob_wages).mean(axis=2)

    prob_obs = prob_obs.reshape(num_obs, state_space.num_types)

    # Accumulate likelihood of the observed choice for each individual-type combination
    # across all periods.
    prob_type = np.multiply.reduceat(prob_obs, rows_start)

    # Multiply each individual-type contribution with its type-specific shares and sum
    # over types to get the likelihood contribution for each individual.
    contribs = (prob_type * type_shares).sum(axis=1)

    return contribs
