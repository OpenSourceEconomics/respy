import numpy as np
from numba import guvectorize

from respy.config import HUGE_FLOAT


def create_draws_and_prob_wages(
    log_wages_observed,
    wages_systematic,
    base_draws,
    choices,
    shocks_cholesky,
    meas_error_sds,
    periods,
):
    """Evaluate likelihood of observed wages and create conditional draws.

    Let nobs be the number of period-individual combinations, i.e. the number of rows
    of the empirical dataset.

    Parameters
    ----------
    log_wages_observed : np.array
        1d numpy array of length nobs * num_types with observed log wages
    wages_systematic : np.array
        2d numpy array of shape (nobs * num_types, nchoices) with systematic wages.
        Can contain np.nan or any number for non-wage sectors. The non wage sectors
        only have to be there to not raise index errors.
    base_draws : np.array
        2d numpy array of shape (ndraws, nchoices) with standard normal random
        random variables.
    choices : np.array
        1d numpy array of length nobs * num_types with the observed choice. Will be used
        to select columns of systematic wages. Therefore it has to be coded starting at
        zero.
    shocks_cholesky : np.array
        1d numpy array of shape (nchoices, nchoices) with the lower triangular cholesky
        factor of the covariance matrix of the shocks.
    meas_error_sds : np.array
        1d numpy array of length (nchoices) with the standard deviations of the
        measurement errors of observed reward components.
    periods : np.array
        1d numpy array of length nobs * num_types with the period of the observation

    Returns
    -------
    draws : np.array
        3d numpy array of shape (nobs * num_types, ndraws, nchoices) with shocks drawn
        from a multivariate normal distribution conditional on the observed wages.
    prob_wages : np.array
        1d numpy array of length nobs * num_types with the unconditional likelihood of
        the observed wages, correcting for measurement error.


    """
    choices = choices.astype(np.uint16)
    nobs, nchoices = wages_systematic.shape
    relevant_systematic_wages = np.choose(choices, wages_systematic.T)
    log_wage_systematic = np.clip(
        np.log(relevant_systematic_wages), -HUGE_FLOAT, HUGE_FLOAT
    )

    states = np.zeros((nobs, nchoices))
    measurements = log_wages_observed - log_wage_systematic
    extended_cholcovs_t = np.zeros((nobs, nchoices + 1, nchoices + 1))
    extended_cholcovs_t[:, 1:, 1:] = shocks_cholesky.T
    meas_sds = meas_error_sds[choices]

    # note: states and extended_cholcovs_t are changed inplace
    prob_wages = kalman_update(
        states, measurements, extended_cholcovs_t, meas_sds, choices
    )

    draws = np.matmul(base_draws[periods], extended_cholcovs_t[:, 1:, 1:])
    draws += states.reshape(nobs, 1, nchoices)
    draws[:, :2] = np.clip(np.exp(draws[:, :2]), 0.0, HUGE_FLOAT)

    return draws, prob_wages


@guvectorize(
    ["f8[:], f8, f8[:, :], f8, u2, f8[:]"], "(nstates), (), (m, m), () ,() -> ()"
)
def kalman_update(state, measurement, extended_cholcov_t, meas_sd, choice, prob_wage):
    """Make a Kalman update and evaluate likelihood of wages.

    extended_cholcov_t  and state are changed in-place for speed reasons.

    Can handle missings in the measurements.

    This is a specialized Kalman filter in the following ways:

    - All measurements just measure one latent factor
    - All factor loadings are equal to 1
    - All initial states are assumed to be equal to 0


    Parameters
    ----------
    state : np.array
        1d array with initial state vector

    measumement : float
        the measurement that is incorporated through the Kalman update

    extended_cholcov_t : np.array
        2d array of shape (nstates + 1, nstates + 1) that contains the transpose of the
        cholesky factor of the state covariance matrix in the lower right block and
        zeros everywhere else.

    meas_sd : float
        The standard deviation of the measurement error

    choice : np.uint16
        Observed choice. Determines on which element of the state vector is measured
        by the measurement.


    Returns
    -------

    prob_wage : float
        The likelihood of the observed wage


    References
    ----------

    Robert Grover Brown. Introduction to Random Signals and Applied Kalman Filtering.
        Wiley and sons, 2012.

    """
    # extract dimensions
    m = len(extended_cholcov_t)
    nfac = m - 1

    # invariant part of the normal pdf
    invariant = 1 / (2 * np.pi) ** 0.5

    # skip missing wages
    if np.isfinite(measurement):
        # construct helper matrix for qr magic
        extended_cholcov_t[0, 0] = meas_sd
        for f in range(1, m):
            extended_cholcov_t[f, 0] = extended_cholcov_t[f, choice + 1]

        # qr decomposition
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

        # likelihood evaluation
        sigma = np.abs(extended_cholcov_t[0, 0])
        prob = invariant / sigma * np.exp(-measurement ** 2 / (2 * sigma ** 2))
        prob_wage[0] = prob

        # state update
        measurement /= sigma
        for f in range(nfac):
            state[f] = extended_cholcov_t[0, f + 1] * measurement
    else:
        prob_wage[0] = 1.0
