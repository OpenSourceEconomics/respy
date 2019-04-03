from numba import guvectorize


@guvectorize(
    [
        "float32[:], float32[:], float32[:], float32[:, :], float32, float32[:]",
        "float64[:], float64[:], float64[:], float64[:, :], float64, float64[:]",
    ],
    "(m), (n), (n), (p, n), () -> ()",
    nopython=True,
    target="parallel",
)
def construct_emax_risk(
    wages, rewards_systematic, emaxs, draws, delta, cont_value
):
    """Simulate expected maximum utility for a given distribution of the unobservables.

    The function takes an agent and calculates the utility for each of the choices, the
    ex-post rewards, with multiple draws from the distribution of unobservables and adds
    the discounted expected maximum utility of subsequent periods resulting from
    choices. Averaging over all maximum utilities yields the expected maximum utility of
    this state.

    The underlying process in this function is called `Monte Carlo integration`_. The
    goal is to approximate an integral by evaluating the integrand at randomly chosen
    points. In this setting, one wants to approximate the expected maximum utility of
    the current state.

    Parameters
    ----------
    wages : np.ndarray
        Array with shape (2,) containing wages.
    rewards_systematic : np.ndarray
        Array with shape (4,) containing systematic rewards.
    emaxs : np.ndarray
        Array with shape (4,) containing expected maximum utility for each choice in the
        subsequent period.
    draws : np.ndarray
        Array with shape (num_draws, 4).
    delta : float
        The discount factor.

    Returns
    -------
    cont_value : float
        Expected maximum utility of an agent.

    .. _Monte Carlo integration:
        https://en.wikipedia.org/wiki/Monte_Carlo_integration

    """
    num_draws, num_choices = draws.shape
    num_wages = wages.shape[0]

    cont_value[0] = 0.0

    for i in range(num_draws):

        current_max_emax = 0.0

        for j in range(num_choices):
            if j < num_wages:
                rew_ex = (
                    wages[j] * draws[i, j] + rewards_systematic[j] - wages[j]
                )
            else:
                rew_ex = rewards_systematic[j] + draws[i, j]

            emax_choice = rew_ex + delta * emaxs[j]

            if emax_choice > current_max_emax:
                current_max_emax = emax_choice

        cont_value[0] += current_max_emax

    cont_value[0] /= num_draws
