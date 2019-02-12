import numpy as np


def construct_emax_risk(
    states, num_draws_emax, period, draws_emax_risk, edu_spec, optim_paras
):
    """ Simulate expected future value for a given distribution of the unobservables.

    Parameters
    ----------
    states : pd.DataFrame
    num_draws_emax : int
    period : int
    draws_emax_risk : np.array
    edu_spec : dict
    optim_paras : dict

    Returns
    -------
    total_values : np.array
        One-dimensional array containing ??? for each state in a given period.

    TODO: This function resembles get_total_values and get_exogenous_variables.
    Refactor!

    """
    states["constant"] = 1.0

    # Construct matrix with dimension (num_states, choices, 1). As edu and home have no
    # wage, initialize positions with ones so that shocks are added.
    wages = states.loc[
        states.period.eq(period), ["wage_a", "wage_b", "constant", "constant"]
    ].values
    wages = wages[:, :, np.newaxis]

    rewards_systematic = states.loc[
        states.period.eq(period),
        [
            "rewards_systematic_a",
            "rewards_systematic_b",
            "rewards_systematic_edu",
            "rewards_systematic_home",
        ],
    ].values
    rewards_systematic = rewards_systematic[:, :, np.newaxis]

    # Combine each states-choices combination with num_draws different draws
    rewards_ex_post = (
        np.multiply(wages, draws_emax_risk.T) + rewards_systematic - wages
    )

    # Calculate total values
    emaxs = states.loc[
        states.period.eq(period),
        ["emaxs_a", "emaxs_b", "emaxs_edu", "emaxs_home"],
    ].values
    emaxs = emaxs[:, :, np.newaxis]

    total_values = rewards_ex_post + optim_paras["delta"] * emaxs

    # Choose maximum value in states and average over draws. Shapes change from
    # (num_states, choices, num_draws) to (num_states, num_draws) to (num_states).
    total_values = total_values.max(axis=1).mean(axis=1)

    return total_values
