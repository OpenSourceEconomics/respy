import numpy as np
import pytest
from numba import jit
from scipy.stats import norm

from respy.conditional_draws import kalman_update


@jit(nopython=True)
def numpy_array_qr(arr):
    """QR decomposition for each matrix in a 3d array."""
    out = np.zeros_like(arr)
    nind = len(arr)
    for i in range(nind):
        q, r = np.linalg.qr(arr[i])
        out[i] = r
    return out


def slow_kalman_update(states, root_covs, measurements, loadings, meas_sd):
    """Make a Kalman update.

    This is a too slow but readable and well tested implementation of a square-root
    Kalman update.

    Params
    ------
    states : np.array
        2d numpy array of shape (nind, nfac) with initial means of the states
    root_covs : np.array
        3d numpy array of shape (nind, nfac, nfac) with lower triangular cholesky
        factors of the state covariance matrix
    measurements : np.array
        1d numpy array of length nind with observed measurements
    loadings : np.array
        1d numpy array of length nfac
    meas_sd : float
        standard deviation of measurement error

    Returns
    -------
    states : np.array
        2d numpy array with updated states
    root_covs : np.array
        3d numpy array with updated covariance matrices

    References
    ----------
    Robert Grover Brown. Introduction to Random Signals and Applied Kalman Filtering.
        Wiley and sons, 2012.

    """
    states = states.copy()
    root_covs = root_covs.copy()

    nobs, nfac = states.shape
    expected_measurements = np.dot(states, loadings)
    residuals = measurements - expected_measurements

    f_stars = np.dot(np.transpose(root_covs, axes=(0, 2, 1)), loadings.reshape(nfac, 1))

    m = np.zeros((nobs, nfac + 1, nfac + 1))
    m[:, 0, 0] = meas_sd
    m[:, 1:, :1] = f_stars
    m[:, 1:, 1:] = np.transpose(root_covs, axes=(0, 2, 1))

    r = numpy_array_qr(m)

    root_covs[:] = np.transpose(r[:, 1:, 1:], axes=(0, 2, 1))

    root_sigmas = r[:, 0, 0]

    kalman_gains = r[:, 0, 1:] / root_sigmas.reshape(nobs, 1)
    states[:] += kalman_gains * residuals.reshape(nobs, 1)

    probs = norm.pdf(residuals, scale=np.abs(r[:, 0, 0]))

    return states, root_covs, probs


def random_kalman_input(seed):
    np.random.seed(seed)
    slow = {}
    nstates = np.random.choice(range(1, 7))
    nind = np.random.choice(range(1, 30))

    slow["states"] = np.zeros((nind, nstates))
    loadings = np.zeros(nstates)
    measured_pos = np.random.choice(range(nstates))
    loadings[measured_pos] = 1.0
    slow["loadings"] = loadings

    slow["measurements"] = np.random.normal(scale=0.2, size=nind)
    slow["meas_sd"] = np.random.uniform(low=0.8, high=1.2)
    root_covs = np.zeros((nind, nstates, nstates))
    for i in range(nind):
        helper = np.eye(nstates)
        helper[np.tril_indices(nstates)] = np.random.uniform(
            low=-0.0001, high=0.00001, size=int(0.5 * nstates * (nstates + 1))
        )
        root_covs[i] = helper
    slow["root_covs"] = root_covs

    choice = np.full(nind, np.argmax(loadings)).astype(np.uint16)

    extended_cholcovs_t = np.zeros((nind, nstates + 1, nstates + 1))
    extended_cholcovs_t[:, 1:, 1:] = np.transpose(root_covs, axes=(0, 2, 1))

    fast = (
        slow["states"],
        slow["measurements"],
        extended_cholcovs_t,
        slow["meas_sd"],
        choice,
    )

    return slow, fast


@pytest.mark.parametrize("seed", range(10))
def test_kalman_update(seed):
    slow_input, fast_input = random_kalman_input(seed)
    slow_states, slow_root_covs, slow_probs = slow_kalman_update(**slow_input)
    fast_probs = kalman_update(*fast_input)
    updated_extended_cholcovs_t = fast_input[2]
    fast_states = fast_input[0]
    fast_root_covs = np.transpose(
        updated_extended_cholcovs_t[:, 1:, 1:], axes=(0, 2, 1)
    )

    slow_covs = np.matmul(slow_root_covs, np.transpose(slow_root_covs, axes=(0, 2, 1)))
    fast_covs = np.matmul(fast_root_covs, np.transpose(fast_root_covs, axes=(0, 2, 1)))

    np.testing.assert_array_almost_equal(fast_states, slow_states)
    np.testing.assert_array_almost_equal(fast_probs, slow_probs)
    np.testing.assert_array_almost_equal(fast_covs, slow_covs)
