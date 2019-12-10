"""Everything related to conditional draws for the maximum likelihood estimation."""
import numpy as np
from estimagic.optimization.utilities import robust_cholesky
from numba import guvectorize

from respy.config import HUGE_FLOAT


def create_draws_and_log_prob_wages(
    log_wage_observed,
    wages_systematic,
    base_draws,
    choices,
    shocks_cholesky,
    n_wages,
    meas_sds,
    is_meas_error,
):
    """Evaluate likelihood of observed wages and create conditional draws.

    Let n_obs be the number of period-individual combinations, i.e. the number of rows
    of the empirical dataset.

    Parameters
    ----------
    log_wage_observed : numpy.ndarray
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
    cov = shocks_cholesky @ shocks_cholesky.T

    updated_means, log_prob_wages = update_mean_and_evaluate_likelihood(
        log_wage_observed, log_wage_systematic, cov, choices, meas_sds
    )

    if is_meas_error:
        updated_chols = update_cholcov_with_measurement_error(
            shocks_cholesky, meas_sds, n_wages
        )
    else:
        updated_chols = update_cholcov(shocks_cholesky, n_wages)

    chol_indices = np.where(np.isfinite(log_wage_observed), choices, n_wages)
    draws = calculate_conditional_draws(
        base_draws, updated_means, updated_chols, chol_indices, HUGE_FLOAT
    )

    return draws, log_prob_wages


@guvectorize(
    ["f8, f8, f8[:, :], u2, f8[:], f8[:], f8[:]"],
    "(), (), (n_choices, n_choices), (), (n_wages) -> (n_choices), ()",
    nopython=True,
)
def update_mean_and_evaluate_likelihood(
    log_wage_observed, log_wage_systematic, cov, choice, meas_sds, updated_mean, loglike
):
    r"""Update mean and evaluate likelihood.

    Calculate the conditional mean of shocks after observing one shock
    and evaluate the likelihood of the observed shock.

    The mean is updated by the "Sequences of Conditional Distributions" explained in
    [1]_. Consider the following sequence of correlated normal random variables whose
    mean is adapted by the following formula:

    .. math::

        X_1 &\sim \mathcal{N}(0, \sigma_{11}) \\
        X_2 &\sim \mathcal{N}(
            \sigma_{12} \frac{X_1}{\sigma_{11}},
            \sigma_{22} - \frac{\sigma^2_{12}}{\sigma_{11}}
        ) \\
        \dots

    For the probability of the observed wage, recognize that wages are log-normally
    distributed. Thus, the following formula applies [2]_ [3]_:

    .. math::

        f_W(w_{it}) = \frac{1}{w_{it}} \cdot \frac{1}{\sigma \sqrt{2 \pi}} \exp \left(
            - \frac{(\ln(w_{it}) - \ln(w(s^-_t, a_t)))^2}{2 \sigma^2}
        \right)

    where :math:`i` is the individual, :math:`t` is the period, :math:`f_W` is the
    probability density function of the wage, :math:`w_{it}` is the observed wage,
    :math:`\sigma` is the standard deviation of the wage shock, :math:`s^-_t` is the
    state without the shocks, :math:`a_t` is the choice and :math:`w(s^-_t, a_t)` is the
    non-stochastic wage implied by the model for choice :math:`a_t`.

    Parameters
    ----------
    log_wage_observed : float
        Log of the observed wage of the individual. Can be ``np.nan`` if no wage was
        observed for a working alternative or the individual chose a non-working
        alternative.
    log_wage_systematic : float
        Log of the implied wage for the choice by the model. This term is computed by
        the wage equation without the choice-specific shock.
    cov : np.ndarray
        Unconditional covariance matrix of the shocks.
    choice : int
        The observed choice.
    meas_sds : np.ndarray
        Array with shape (n_choices,) containing standard errors of measurement errors.

    Returns
    -------
    updated_mean : np.ndarray
        Conditional mean of shocks, given the observed shock. Contains the observed
        shock in the corresponding position even in the degenerate case of no
        measurement error. Has length n_choices.
    loglike : float
        log likelihood of observing the observed shock. 0 if no shock was observed.

    References
    ----------
    .. [1] Gentle, J. E. (2009). Computational statistics (Vol. 308). New York:
           Springer.
    .. [2] Johnson, Norman L.; Kotz, Samuel; Balakrishnan, N. (1994), "14: Lognormal
           Distributions", Continuous univariate distributions. Vol. 1, Wiley Series in
           Probability and Mathematical Statistics: Applied Probability and Statistics
           (2nd ed.)
    .. [3] Keane, M. P., Wolpin, K. I., & Todd, P. (2011). Handbook of Labor Economics,
           volume 4, chapter The Structural Estimation of Behavioral Models: Discrete
           Choice Dynamic Programming Methods and Applications.

    """
    n_choices = len(cov)
    invariant = -np.log(2 * np.pi) / 2
    shock = log_wage_observed - log_wage_systematic

    if np.isfinite(shock):
        sigma_squared = cov[choice, choice] + meas_sds[choice] ** 2
        sigma = np.sqrt(sigma_squared)
        for i in range(n_choices):
            updated_mean[i] = cov[choice, i] * shock / sigma_squared
        loglike[0] = (
            invariant
            - log_wage_observed
            - np.log(sigma)
            - shock ** 2 / (2 * sigma_squared)
        )
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

        r = np.linalg.qr(extended_cholcov_t, mode="r")
        updated_chols[i] = make_cholesky_unique(r[1:, 1:].T)

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

    for i in range(n_wages):
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
def calculate_conditional_draws(
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


def make_cholesky_unique(chol):
    """Make a lower triangular cholesky factor unique.

    Cholesky factors are only unique with the additional requirement that all diagonal
    elements are positive. This is done automatically by np.linalg.cholesky.
    Since we calucate cholesky factors by QR decompositions we have to do it manually.

    It is obvious from that this is admissible because:

    chol sign_swither sign_switcher.T chol.T = chol chol.T

    """
    sign_switcher = np.sign(np.diag(np.diagonal(chol)))
    return chol @ sign_switcher
