import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from respy.conditional_draws import calculate_conditional_draws
from respy.conditional_draws import update_cholcov
from respy.conditional_draws import update_cholcov_with_measurement_error
from respy.conditional_draws import update_mean_and_evaluate_likelihood
from respy.config import MAX_FLOAT
from respy.config import TEST_RESOURCES_DIR


@pytest.fixture
def kalman_results():
    """The inputs and outputs were generated using a well tested Kalman filter."""
    with open(TEST_RESOURCES_DIR / "conditional_draws_fixture.pickle", "rb") as p:
        fix = pickle.load(p)
    return fix


@pytest.mark.parametrize("i", range(20))
def test_update_and_evaluate_likelihood(i, kalman_results):
    inp = kalman_results["mean"][i]["input"]
    calculated_mean, calculated_like = update_mean_and_evaluate_likelihood(*inp)
    expected_mean = kalman_results["mean"][i]["output_mean"]
    expected_like = kalman_results["mean"][i]["output_loglike"]
    aaae(calculated_mean, expected_mean)
    aaae(calculated_like, expected_like)


@pytest.mark.parametrize("i", range(10))
def test_update_cholcovs_with_error(i, kalman_results):
    inp = kalman_results["cov_error"][i]["input"]
    calculated_chol = update_cholcov_with_measurement_error(**inp)
    expected_chol = kalman_results["cov_error"][i]["output"]

    calculated_cov = np.matmul(
        calculated_chol, np.transpose(calculated_chol, axes=(0, 2, 1))
    )
    expected_cov = np.matmul(expected_chol, np.transpose(expected_chol, axes=(0, 2, 1)))

    aaae(calculated_cov, expected_cov)


def test_update_cholcovs():
    cov = np.array([[1, 0.8, 0.8], [0.8, 1, 0.8], [0.8, 0.8, 1]])

    chol = np.linalg.cholesky(cov)

    expected = np.array(
        [
            [[0, 0, 0], [0, 0.6, 0], [0, 0.26666667, 0.53748385]],
            [[0.6, 0, 0], [0, 0, 0], [0.26666667, 0, 0.53748385]],
            [[1, 0, 0], [0.8, 0.6, 0], [0.8, 0.26666667, 0.53748385]],
        ]
    )

    calculated = update_cholcov(chol, 2)
    i = 1
    aaae(calculated[i], expected[i], decimal=5)


def test_calculate_conditional_draws():
    draws = np.array([[0.5, -1, 1]])
    updated_mean = np.arange(3)
    updated_chols = np.zeros((3, 3, 3))
    updated_chols[1] = np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])

    calculated = calculate_conditional_draws(
        draws, updated_mean, updated_chols, 1, MAX_FLOAT
    )[0]
    expected = np.array([1.64872127, 0.36787944, 5])

    aaae(calculated, expected)
