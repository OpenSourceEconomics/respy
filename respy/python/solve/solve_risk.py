from respy.python.shared.shared_auxiliary import get_continuation_value


def construct_emax_risk(rewards, emaxs, draws_emax_risk, optim_paras):
    """ Simulate expected future value for a given distribution of the unobservables.

    Note that, this function works on all states for a given period.

    Parameters
    ----------
    rewards : np.ndarray
        Array with shape (num_states_in_period, 9).
    emaxs : np.ndarray
        Array with shape (num_states_in_period, 4).
    draws_emax_risk : np.array
    optim_paras : dict

    Returns
    -------
    total_values : np.array
        One-dimensional array containing ??? for each state in a given period.

    """
    total_values, _ = get_continuation_value(
        rewards[:, -2:],
        rewards[:, :4],
        draws_emax_risk,
        emaxs,
        optim_paras["delta"],
    )

    # Choose maximum value in states and average over draws. Shape changes from
    # (num_states, choices, num_draws) to (num_states, num_draws) to (num_states).
    total_values = total_values.max(axis=1).mean(axis=1)

    return total_values
