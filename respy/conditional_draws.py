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
    meas_sds,
    is_meas_error,
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
    periods : numpy.ndarray
        Array with shape (n_obs * n_types,) containing the period of the observation.
    n_wages : int
        Number of wage sectors
    meas_sds : numpy.ndarray
        Array with shape (n_choices,) containing standard deviations of the measurement
        errors of observed reward components. It is 0 for choices where no reward
        component is observed.
    is_meas_error : bool

    Returns
    -------
    draws : numpy.ndarray
        Array with shape (n_obs * n_types, n_draws, n_choices) containing shocks drawn
        from a multivariate normal distribution conditional on the observed wages.
    log_prob_wages : numpy.ndarray
        Array with shape (n_obs * n_types,) containing the unconditional log likelihood
        of the observed wages, correcting for measurement error if necessary.

    """
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

    if is_meas_error:
        updated_chols = update_cholcov_with_measurement_error(
            shocks_cholesky, meas_sds, n_wages
        )
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
    cov : np.ndarray
        Unconditional covariance matrix of the shocks
    choice : int
        The observed choice
    meas_sds : np.ndarray
        1d array of length n_choices with standard errors of measurement errors.

    Returns
    -------
    updated_mean : np.ndarray
        Conditional mean of shocks, given the observed shock. Contains the observed
        shock in the corresponding position even in the degenerate case of no
        measurement error. Has length n_choices.
    loglike : float
        log likelihood of observing the observed shock. 0 if no shock was observed.

    """
    n_choices = len(cov)
    invariant = np.log(1 / (2 * np.pi) ** 0.5)

    if np.isfinite(shock):
        sigma_squared = cov[choice, choice] + meas_sds[choice] ** 2
        sigma = np.sqrt(sigma_squared)
        for i in range(n_choices):
            updated_mean[i] = cov[choice, i] * shock / sigma_squared
        loglike[0] = invariant - np.log(sigma) - shock ** 2 / (2 * sigma_squared)
    else:
        for i in range(n_choices):
            updated_mean[i] = 0
        loglike[0] = 0


def update_cholcov_with_measurement_error(shocks_cholesky, meas_sds, n_wages):
    """Make a Kalman covariance updated for all possible cases.

    Parameters
    ----------
    shocks_cholesky : numpy.ndarray
        cholesky factor of the covariance matrix before updating. Has
        dimension (n_choices, n_choices)

    meas_sds: numpy.ndarray
        the standard deviations of the measurement errors. Has length n_wages.

    n_wages : int
        number of wage sectors.

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
    n_choices = len(shocks_cholesky)

    updated_chols = np.zeros((n_wages + 1, n_choices, n_choices))

    for i in range(n_wages):
        extended_cholcov_t = np.zeros((n_choices + 1, n_choices + 1))
        extended_cholcov_t[1:, 1:] = shocks_cholesky.T
        extended_cholcov_t[0, 0] = meas_sds[i]
        extended_cholcov_t[1:, 0] = shocks_cholesky.T[:, i]

        m = n_choices + 1
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

        updated_chols[i] = extended_cholcov_t[1:, 1:].T
    updated_chols[-1] = shocks_cholesky
    return updated_chols


def update_cholcov(shocks_cholesky, n_wages):
    """Calculate cholesky factors of conditional covs for all possible cases.

    Parameters
    ----------
    shocks_cholesky : numpy.ndarray
        cholesky factor of the covariance matrix before updating. Has
        dimension (n_choices, n_choices)
    n_wages : int
        Number of wage sectors.

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

    for i in range(n_choices):
        reduced_cov = np.delete(np.delete(cov, i, axis=1), i, axis=0)
        choice_var = cov[i, i]

        f = np.delete(cov[i], i)

        updated_reduced_cov = reduced_cov - np.outer(f, f) / choice_var
        updated_reduced_chol = robust_cholesky(updated_reduced_cov)

        updated_chols[i, :i, :i] = updated_reduced_chol[:i, :i]
        updated_chols[i, :i, i + 1 :] = updated_reduced_chol[:i, i:]
        updated_chols[i, i + 1 :, :i] = updated_reduced_chol[i:, :i]
        updated_chols[i, i + 1 :, i + 1 :] = updated_reduced_chol[i:, i:]

    updated_chols[-1] = shocks_cholesky

    return updated_chols


@guvectorize(
    ["f8[:, :], f8[:], f8[:, :, :], u2, f8, f8[:, :]"],
    "(n_draws, n_choices), (n_choices), (n_wages_plus_one, n_choices, n_choices), (), "
    "() -> (n_draws, n_choices)",
    nopython=True,
)
def calculate_conditional_draws2(
    base_draws, updated_mean, updated_chols, chol_index, huge_float, conditional_draw
):
    """Calculate the conditional draws from base draws, updated means and updated chols.

    Parameters
    ----------
    base_draws : np.ndarray
        iid standard normal draws
    updated_mean : np.ndarray
        conditional mean, given the observed shock. Contains the observed shock in the
        corresponding position.
    updated_chols : np.ndarray
        cholesky factor of conditional covariance, given the observed shock. If there is
        no measurement error, it contains a zero column and row at the position of the
        observed shock.
    chol_index : float
        index of the relevant updated cholesky factor
    huge_float : float
        value at which exponentials are clipped.

    Returns
    -------
    conditional draws : np.ndarray
        draws from the conditional distribution of the shocks.

    """
    n_draws, n_choices = base_draws.shape
    n_wages = len(updated_chols) - 1

    for d in range(n_draws):
        for i in range(n_choices):
            cd = updated_mean[i]
            for j in range(i + 1):
                cd += base_draws[d, j] * updated_chols[chol_index, i, j]
            if i < n_wages:
                cd = np.exp(cd)
                if cd > huge_float:
                    cd = huge_float
            conditional_draw[d, i] = cd


def calculate_conditional_draws(base_draws, updated_mean, updated_chols, chol_index):
    """This function replicates a bug we had previously to make regression run."""
    n_wages = len(updated_chols) - 1
    n_obs, n_choices = updated_mean.shape
    chols_t = np.transpose(updated_chols[chol_index], axes=(0, 2, 1))
    draws = np.matmul(base_draws, chols_t)
    draws += updated_mean.reshape(n_obs, 1, n_choices)
    # This is wrong
    draws[:, :n_wages] = np.clip(np.exp(draws[:, :n_wages]), 0.0, HUGE_FLOAT)

    return draws
