import numpy as np
import pytest
from scipy.special import softmax

from respy.shared import predict_multinomial_logit


@pytest.mark.parametrize(
    "coefficients, type_proportions",
    [
        (
            np.array(
                [[0.0, 0.0], [2.4356, 0.3136], [2.5362, 1.0523], [-0.1598, -0.7369]]
            ),
            ([0.1751, 0.2396, 0.5015, 0.0838], [0.0386, 0.4409, 0.4876, 0.0329]),
        ),
        (
            np.array([[0.0, 0.0], [-0.0034, 1.3979], [0.466, 2.114], [-0.389, 1.9514]]),
            ([0.0491, 0.1987, 0.4066, 0.3456], [0.2343, 0.2335, 0.3734, 0.1588]),
        ),
    ],
)
def test_predict_multinomial_logit(coefficients, type_proportions):
    result = predict_multinomial_logit(coefficients, [[0, 1], [1, 0]])

    result_ = softmax(coefficients.dot([[0, 1], [1, 0]]), axis=0).T

    np.testing.assert_allclose(result, result_)

    np.testing.assert_allclose(type_proportions, result, rtol=1e-4)
