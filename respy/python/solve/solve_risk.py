import numpy as np

from numba import guvectorize


@guvectorize(
    [
        "float32[:], float32[:], float32[:], float32[:, :], float32, float32[:]",
        "float64[:], float64[:], float64[:], float64[:, :], float64, float64[:]",
    ],
    "(m), (n), (n), (p, n), () -> ()",
    nopython=True,
    target="cpu",
)
def construct_emax_risk(
    wages, rewards_systematic, emaxs_sub_period, draws, delta, cont_value
):
    """Simulate expected future value for a given distribution of the unobservables.

    Note that, this function is a guvectorized function which is flexible in the number
    of individuals and number of draws. It is also related to
    :func:`get_continuation_value`, but also includes an reduction by taking the mean
    over maximum utilities per draw.

    Parameters
    ----------
    wages : np.ndarray
        Array with shape (num_states_in_period, 2).
    rewards : np.ndarray
        Array with shape (num_states_in_period, 4).
    emaxs : np.ndarray
        Array with shape (num_states_in_period, 4).
    draws_emax_risk : np.ndarray
        Array with shape (num_draws, 4).
    delta : np.ndarray
        Scalar value representing the discount factor.

    Returns
    -------
    total_values : np.ndarray
        Array with shape (num_states_in_period,) containing emax for each agent in a
        period.

    Warning
    -------
    Note, the funny syntax if guvectorize should return a scalar value
    (https://github.com/numba/numba/issues/2935). This might change in the future.

    """
    num_draws = draws.shape[0]
    cont_value[0] = 0.0

    for i in range(num_draws):

        current_max_emax = 0.0

        for j in range(4):
            if j < 2:
                rew_ex = (
                    wages[j] * draws[i, j] + rewards_systematic[j] - wages[j]
                )
            else:
                rew_ex = rewards_systematic[j] + draws[i, j]

            emax_choice = rew_ex + delta * emaxs_sub_period[j]

            if emax_choice > current_max_emax:
                current_max_emax = emax_choice

        cont_value[0] += current_max_emax

    cont_value[0] /= num_draws
