from respy.python.shared.shared_auxiliary import get_continuation_value


def construct_emax_risk(states, draws_emax_risk, optim_paras):
    """ Simulate expected future value for a given distribution of the unobservables.

    Parameters
    ----------
    states : pd.DataFrame
    draws_emax_risk : np.array
    optim_paras : dict

    Returns
    -------
    total_values : np.array
        One-dimensional array containing ??? for each state in a given period.

    """

    total_values, _ = get_continuation_value(
        states[["wage_a", "wage_b"]].values,
        states[
            [
                "rewards_systematic_a",
                "rewards_systematic_b",
                "rewards_systematic_edu",
                "rewards_systematic_home",
            ]
        ].values,
        draws_emax_risk,
        states[["emaxs_a", "emaxs_b", "emaxs_edu", "emaxs_home"]].values,
        optim_paras["delta"]
    )

    # Choose maximum value in states and average over draws. Shapes change from
    # (num_states, choices, num_draws) to (num_states, num_draws) to (num_states).
    total_values = total_values.max(axis=1).mean(axis=1)

    return total_values
