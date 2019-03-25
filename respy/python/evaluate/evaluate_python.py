import numpy as np

from respy.python.shared.shared_auxiliary import get_conditional_probabilities
from respy.python.evaluate.evaluate_auxiliary import (
    get_smoothed_probability,
    get_pdf_of_normal_distribution,
)
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.python.shared.shared_auxiliary import get_continuation_value
from numba import guvectorize


def pyth_contributions(
    state_space,
    data,
    periods_draws_prob,
    tau,
    num_agents_est,
    num_obs_agent,
    optim_paras,
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
    wages = data["Wage"].values

    # Extend systematic wages with zeros so that indexing with choice three and four
    # does not fail.
    wages_systematic_ext = np.hstack(
        (state_space.rewards[:, -2:], np.zeros((state_space.num_states, 2)))
    )

    # Define a shortcut to the Cholesky factors of the shocks, because they are so
    # frequently used inside deeply nested loops.
    sc = optim_paras["shocks_cholesky"]

    # Initialize auxiliary objects.
    contribs = np.full(num_agents_est, np.nan)

    # Calculate the probability for all numbers of observations. The format of array is
    # (num_obs, num_types, num_draws, num_choices) reduced to the minimum of dimensions.
    # E.g. choices are determined for every agent thus the shape is (num_obs,),
    # prob_wages are calculated for each draw thus the shape is (num_obs, num_types,
    # num_draws).
    rows_start = np.hstack((0, np.cumsum(num_obs_agent)[:-1]))
    edu_starts = agents[rows_start, 3]

    # Update type probabilities conditional on edu_start > 9.
    type_shares = get_conditional_probabilities(
        optim_paras["type_shares"], edu_starts
    )

    num_obs = num_obs_agent.sum()

    # Extract observable components of the state space as well as agent's
    # decision.
    periods, exp_as, exp_bs, edus, choices_lagged, choices = (
        agents[:num_obs, i] for i in range(6)
    )

    # Get state index to access the systematic component of the agents
    # rewards. These feed into the simulation of choice probabilities.
    ks = state_space.indexer[
        periods, exp_as, exp_bs, edus, choices_lagged - 1, :
    ]

    wages_observed = wages[:num_obs].reshape(-1, 1)

    wages_systematic = wages_systematic_ext[ks, choices.reshape(-1, 1) - 1]

    # Calculate the disturbance which are implied by the model and the observed wages.
    dist = np.clip(np.log(wages_observed), -HUGE_FLOAT, HUGE_FLOAT) - np.clip(
        np.log(wages_systematic), -HUGE_FLOAT, HUGE_FLOAT
    )
    dist = dist.reshape(num_obs, state_space.num_types, 1)

    # Get periods based on the shape (num_obs, num_types) of the indexer.
    periods = state_space.states[ks, 0]

    # Extract relevant deviates from standard normal distribution. The same set of
    # baseline draws are used for each agent and period. The resulting shape is
    # (num_obs, num_types, num_draws, num_choices).
    draws_stan = np.take(periods_draws_prob, periods, axis=0)

    # Initialize prob_wages with ones as it is the value for SCHOOLING and HOME and
    # OCCUPATIONS without wages.
    prob_wages = np.ones((num_obs, state_space.num_types, num_draws_prob))

    # If an agent is observed working, then the the labor market shocks are observed and
    # the conditional distribution is used to determine the choice probabilities if the
    # wage information is available as well.
    is_working = np.isin(choices, [1, 2])
    has_wage = ~np.isnan(wages_observed).ravel()
    # These are the indices for observations where shocks and probabilities of wages
    # need to be adjusted.
    idx_choice_1 = np.where((choices == 1) & is_working & has_wage)
    idx_choice_2 = np.where((choices == 2) & is_working & has_wage)

    # Adjust draws and prob_wages in cases of OCCUPATION A with wages.
    if not idx_choice_1[0].shape[0] == 0:
        draws_stan[idx_choice_1, :, :, 0] = dist[idx_choice_1] / sc[0, 0]
        prob_wages[idx_choice_1] = get_pdf_of_normal_distribution(
            dist[idx_choice_1], 0, sc[0, 0]
        )

    # Adjust draws and prob_wages in cases of OCCUPATION B with wages.
    if not idx_choice_2[0].shape[0] == 0:
        draws_stan[idx_choice_2, :, :, 1] = (
            dist[idx_choice_2] - sc[1, 0] * draws_stan[idx_choice_2, :, :, 0]
        ) / sc[1, 1]
        means = sc[1, 0] * draws_stan[idx_choice_2, :, :, 0]
        prob_wages[idx_choice_2] = get_pdf_of_normal_distribution(
            dist[idx_choice_2], means, sc[1, 1]
        )

    draws = np.tensordot(draws_stan, sc.T, axes=(3, 1))

    draws[:, :, :, :2] = faster_exp_clip(draws[:, :, :, :2])

    total_values = get_continuation_value(
        state_space.rewards[ks, -2:],
        state_space.rewards[ks, :4],
        state_space.emaxs[ks, :4],
        draws,
        optim_paras["delta"],
    )

    total_values = total_values.transpose(0, 1, 3, 2)

    prob_choices = get_smoothed_probability(
        total_values, choices.reshape(-1, 1) - 1, tau
    )

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


@guvectorize(
    ["float64[:], float64[:]"], "(n) -> (n)", nopython=True, target="parallel"
)
def faster_exp_clip(x, y):
    for i in range(x.shape[0]):
        temp = np.exp(x[i])
        if temp < 0:
            y[i] = 0
        elif temp > HUGE_FLOAT:
            y[i] = HUGE_FLOAT
        else:
            y[i] = temp
