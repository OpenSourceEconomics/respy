import numpy as np
from estimagic.optimization.utilities import robust_cholesky
from numba import guvectorize

from respy.config import HUGE_FLOAT


def create_draws_and_log_prob_wages(
    log_wages_observed,
    wages_systematic,
    base_draws,
    choices,
    shocks_cholesky,
    periods,
    n_wages,
    meas_sds=None,
):
    """Evaluate likelihood of observed wages and create conditional draws.

    Let n_obs be the number of period-individual combinations, i.e. the number of rows
    of the empirical dataset.

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
    meas_sds : numpy.ndarray
        Array with shape (n_wages,) containing standard deviations of the measurement
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
    if meas_sds is None:
        meas_sds = np.zeros(n_wages)
        meas_error = False
    else:
        meas_sds = meas_sds[:n_wages]
        meas_error = True

    n_obs, n_choices = wages_systematic.shape

    choices = choices.astype(np.uint16)
    relevant_systematic_wages = np.choose(choices, wages_systematic.T)
    log_wage_systematic = np.clip(
        np.log(relevant_systematic_wages), -HUGE_FLOAT, HUGE_FLOAT
    )
    observed_shocks = log_wages_observed - log_wage_systematic
    cov = shocks_cholesky @ shocks_cholesky.T

    updated_means, log_prob_wages = update_mean_and_evaluate_likelihood(
        observed_shocks, cov, choices, meas_sds
    )

    if meas_error:
        updated_chols = update_cholcov_with_measurement_error(shocks_cholesky, meas_sds)
    else:
        updated_chols = update_cholcov(shocks_cholesky, n_wages)

    chol_indices = np.where(np.isfinite(log_wages_observed), choices, n_wages)
    draws = calculate_conditional_draws(
        base_draws[periods], updated_means, updated_chols, chol_indices
    )

    return draws, log_prob_wages


@guvectorize(
    ["f8, f8[:, :], u2, f8[:], f8[:], f8[:]"],
    "(), (n_choices, n_choices), (), (n_wages) -> (n_choices), ()",
    nopython=True,
)
def update_mean_and_evaluate_likelihood(
    shock, cov, choice, meas_sds, updated_mean, loglike
):
    """Update mean and evaluate likelihood.

    Calculate the conditional mean of shocks after observing one shock
    and evaluate the likelihood of the observed shock.

    Parameters
    ----------
    shock : float
        The observed shock. NaN for individuals where no shock was observed.


    Returns
    -------
    updated_mean : np.ndarray
        Conditional mean of shocks, given the observed shock. Contains the observed
        shock in the corresponding position even in the degenerate case of no
        measurement error. Has length n_choices.
    loglike : float
        log likelihood of observing the observed shock. 0 if no shock was observed.


    """
    dim = len(cov)
    invariant = np.log(1 / (2 * np.pi) ** 0.5)

    if np.isfinite(shock):
        sigma_squared = cov[choice, choice] + meas_sds[choice] ** 2
        sigma = np.sqrt(sigma_squared)
        for i in range(dim):
            updated_mean[i] = cov[choice, i] * shock / sigma_squared
        loglike[0] = invariant - np.log(sigma) - shock ** 2 / (2 * sigma_squared)
    else:
        for i in range(dim):
            updated_mean[i] = 0
        loglike[0] = 0


def update_cholcov_with_measurement_error(shocks_cholesky, meas_sds):
    """Make a Kalman covariance updated for all possible cases.

    Parameters
    ----------
    shocks_cholesky : numpy.ndarray
        cholesky factor of the covariance matrix before updating. Has
        dimension (n_choices, n_choices)

    meas_sds: numpy.ndarray
        the standard deviations of the measurement errors. Has length n_wages.

    Returns
    -------
    updated_chols : numpy.ndarray
        Array of (shape n_wages + 1, n_choices, n_choices) with the cholesky factors
        of the updated covariance matrices for each possible observed shock. The
        last element corresponds to not observing any shock.


    We use a square-root implementation of the Kalman filter to avoid taking any
    Cholesky decompositions which could fail due to numerical error.

    References
    ----------
    Robert Grover Brown. Introduction to Random Signals and Applied Kalman Filtering.
        Wiley and sons, 2012.

    """
    n_wages = len(meas_sds)
    n_choices = len(shocks_cholesky)

    updated_chols = np.zeros((n_wages + 1, n_choices, n_choices))

    for choice in range(n_wages):
        extended_cholcov_t = np.zeros((n_choices + 1, n_choices + 1))
        extended_cholcov_t[1:, 1:] = shocks_cholesky.T
        extended_cholcov_t[0, 0] = meas_sds[choice]
        extended_cholcov_t[1:, 0] = shocks_cholesky.T[:, choice]
        r = np.linalg.qr(extended_cholcov_t, mode="r")
        updated_chols[choice] = r[1:, 1:].T

    updated_chols[-1] = shocks_cholesky
    return updated_chols


def update_cholcov(shocks_cholesky, n_wages):
    """Calculate cholesky factors of conditional covs for all possible cases.

    Parameters
    ----------
    shocks_cholesky : numpy.ndarray
        cholesky factor of the covariance matrix before updating. Has
        dimension (n_choices, n_choices)

    Returns
    -------
    updated_chols : numpy.ndarray
        Array of (shape n_wages + 1, n_choices, n_choices) with the cholesky factors
        of the updated covariance matrices for each possible observed shock. The
        last element corresponds to not observing any shock.

    """
    n_choices = len(shocks_cholesky)
    cov = shocks_cholesky @ shocks_cholesky.T

    updated_chols = np.zeros((n_wages + 1, n_choices, n_choices))

    for choice in range(n_choices):
        reduced_cov = np.delete(np.delete(cov, choice, axis=1), choice, axis=0)
        choice_var = cov[choice, choice]

        f = np.delete(cov[choice], choice)

        updated_reduced_cov = reduced_cov - np.outer(f, f) / choice_var
        updated_reduced_chol = robust_cholesky(updated_reduced_cov)

        updated_chols[choice, :choice, :choice] = updated_reduced_chol[:choice, :choice]
        updated_chols[choice, :choice, choice + 1 :] = updated_reduced_chol[
            :choice, choice:
        ]
        updated_chols[choice, choice + 1 :, :choice] = updated_reduced_chol[
            choice:, :choice
        ]
        updated_chols[choice, choice + 1 :, choice + 1 :] = updated_reduced_chol[
            choice:, choice:
        ]

    updated_chols[-1] = shocks_cholesky

    return updated_chols


@guvectorize(
    ["f8[:, :], f8[:], f8[:, :, :], u2, f8[:, :]"],
    "(n_draws, n_choices), (n_choices), (n_wages_plus_one, n_choices, n_choices), () "
    "-> (n_draws, n_choices)",
)
def calculate_conditional_draws(
    draws, updated_mean, updated_chols, chol_index, conditional_draw
):
    """Calculate the conditional draws from base draws, updated means and updated chols.

    """
    n_draws, n_choices = draws.shape
    n_wages = len(updated_chols) - 1

    for d in range(n_draws):
        for i in range(n_choices):
            cd = updated_mean[i]
            for j in range(i + 1):
                cd += draws[d, j] * updated_chols[chol_index, i, j]
            if i < n_wages:
                cd = np.exp(cd)
                if cd > HUGE_FLOAT:
                    cd = HUGE_FLOAT
            conditional_draw[d, i] = cd
