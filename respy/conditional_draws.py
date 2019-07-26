import numpy as np
from numba import guvectorize

from respy.config import HUGE_FLOAT


def create_draws_and_log_prob_wages(
    log_wages_observed,
    wages_systematic,
    base_draws,
    choices,
    shocks_cholesky,
    meas_error_sds,
    periods,
    n_wages,
):
    """Evaluate likelihood of observed wages and create conditional draws.

    Let n_obs be the number of period-individual combinations, i.e. the number of rows
    of the empirical dataset.

    Note
    ----
    This function calls :func:`kalman_update` which changes ``states`` and
    ``extended_cholcovs_t`` in-place.

    Parameters
    ----------
    log_wages_observed : numpy.ndarray
        Array with shape (n_obs * n_types,) containing observed log wages.
    wages_systematic : numpy.ndarray
        Array with shape (n_obs * n_types, n_choices) containing systematic wages. Can
        contain numpy.nan or any number for non-wage choices. The non-wage choices only
        have to be there to not raise index errors.
    base_draws : numpy.ndarray
        Array with shape (n_draws, n_choices) with standard normal random variables.
    choices : numpy.ndarray
        Array with shape (n_obs * n_types,) containing observed choices. Is used to
        select columns of systematic wages. Therefore it has to be coded starting at
        zero.
    shocks_cholesky : numpy.ndarray
        Array with shape (n_choices, n_choices) with the lower triangular Cholesky
        factor of the covariance matrix of the shocks.
    meas_error_sds : numpy.ndarray
        Array with shape (n_choices,) containing standard deviations of the measurement
        errors of observed reward components.
    periods : numpy.ndarray
        Array with shape (n_obs * n_types,) containing the period of the observation.

    Returns
    -------
    draws : numpy.ndarray
        Array with shape (n_obs * n_types, n_draws, n_choices) containing shocks drawn
        from a multivariate normal distribution conditional on the observed wages.
    log_prob_wages : numpy.ndarray
        Array with shape (n_obs * n_types,) containing the unconditional log likelihood
        of the observed wages, correcting for measurement error.

    """
    choices = choices.astype(np.uint16)
    n_obs, n_choices = wages_systematic.shape
    relevant_systematic_wages = np.choose(choices, wages_systematic.T)
    log_wage_systematic = np.clip(
        np.log(relevant_systematic_wages), -HUGE_FLOAT, HUGE_FLOAT
    )

    states = np.zeros((n_obs, n_choices))
    measurements = log_wages_observed - log_wage_systematic
    extended_cholcovs_t = np.zeros((n_obs, n_choices + 1, n_choices + 1))
    extended_cholcovs_t[:, 1:, 1:] = shocks_cholesky.T
    meas_sds = meas_error_sds[choices]

    # Note: ``states`` and ``extended_cholcovs_t`` are changed in-place.
    log_prob_wages = kalman_update(
        states, measurements, extended_cholcovs_t, meas_sds, choices
    )

    draws = np.matmul(base_draws[periods], extended_cholcovs_t[:, 1:, 1:])
    draws += states.reshape(n_obs, 1, n_choices)
    draws[:, :n_wages] = np.clip(np.exp(draws[:, :n_wages]), 0.0, HUGE_FLOAT)

    return draws, log_prob_wages


@guvectorize(
    ["f8[:], f8, f8[:, :], f8, u2, f8[:]"],
    "(n_states), (), (n_choices_plus_one, n_choices_plus_one), (), () -> ()",
)
def kalman_update(
    state, measurement, extended_cholcov_t, meas_sd, choice, log_prob_wage
):
    """Make a Kalman update and evaluate log likelihood of wages.

    This function is a specialized Kalman filter in the following ways:

    - All measurements just measure one latent factor
    - All factor loadings are equal to 1
    - All initial states are assumed to be equal to 0

    ``extended_cholcov_t`` and ``state`` are changed in-place for performance reasons.

    The function can handle missings in the measurements.

    Parameters
    ----------
    state : numpy.ndarray
        Array with shape (n_choices,) containing initial state vectors.
    measurement : float
        The measurement that is incorporated through the Kalman update.
    extended_cholcov_t : numpy.ndarray
        Array with shape (n_states + 1, n_states + 1) that contains the transpose of the
        Cholesky factor of the state covariance matrix in the lower right block and
        zeros everywhere else.
    meas_sd : float
        The standard deviation of the measurement error.
    choice : numpy.uint16
        Observed choice. Determines on which element of the state vector is measured
        by the measurement.

    Returns
    -------
    log_prob_wage : float
        The log likelihood of the observed wage.

    References
    ----------
    Robert Grover Brown. Introduction to Random Signals and Applied Kalman Filtering.
        Wiley and sons, 2012.

    """
    # Extract dimensions.
    m = len(extended_cholcov_t)
    nfac = m - 1

    # This is the invariant part of a normal probability density function.
    invariant = np.log(1 / (2 * np.pi) ** 0.5)

    # Skip missing wages.
    if np.isfinite(measurement):
        # Construct helper matrix for qr magic.
        extended_cholcov_t[0, 0] = meas_sd
        for f in range(1, m):
            extended_cholcov_t[f, 0] = extended_cholcov_t[f, choice + 1]

        # QR decomposition.
        for f in range(m):
            for g in range(m - 1, f, -1):
                b = extended_cholcov_t[g, f]
                if b != 0.0:
                    a = extended_cholcov_t[g - 1, f]
                    if abs(b) > abs(a):
                        r_ = a / b
                        s_ = 1 / (1 + r_ ** 2) ** 0.5
                        c_ = s_ * r_
                    else:
                        r_ = b / a
                        c_ = 1 / (1 + r_ ** 2) ** 0.5
                        s_ = c_ * r_
                    for k_ in range(m):
                        helper1 = extended_cholcov_t[g - 1, k_]
                        helper2 = extended_cholcov_t[g, k_]
                        extended_cholcov_t[g - 1, k_] = c_ * helper1 + s_ * helper2
                        extended_cholcov_t[g, k_] = -s_ * helper1 + c_ * helper2

        # Likelihood evaluation.
        sigma = np.abs(extended_cholcov_t[0, 0])
        prob = invariant - np.log(sigma) - measurement ** 2 / (2 * sigma ** 2)
        log_prob_wage[0] = prob

        # Update the state.
        measurement /= sigma
        for f in range(nfac):
            state[f] = extended_cholcov_t[0, f + 1] * measurement
    else:
        log_prob_wage[0] = 0
