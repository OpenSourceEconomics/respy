"""Test the likelihood routine."""
import numpy as np

from respy.likelihood import create_draws_and_prob_wages


def test_create_draws_and_prob_wages():
    """This test ensures that the matrix multiplication returns the correct result.

    The problem is that the second part of the function should yield a matrix
    multiplication of ``draws.dot(sc.T)``, but ``sc`` is not transposed. We will run the
    function with appropriate arguments and receive ``draws`` as the first return value.
    Then, we will reverse the matrix multiplication with ``temp =
    draws.dot(np.inv(sc.T))`` and redo it with ``result = temp.dot(sc.T)``. If ``draws
    == result``, the result is correct.

    """
    wage_observed = 2.5
    wage_systematic = np.array([2.0, 2.0])
    period = 0
    draws = np.random.randn(1, 20, 4)
    choice = 2
    sc = np.array(
        [
            [0.43536991, 0.15, 0.0, 0.0],
            [0.15, 0.48545471, 0.0, 0.0],
            [0.0, 0.0, 0.09465536, 0.0],
            [0.0, 0.0, 0.0, 0.46978499],
        ]
    )

    draws, _ = create_draws_and_prob_wages(
        wage_observed, wage_systematic, period, draws, choice, sc
    )

    temp = draws.dot(np.linalg.inv(sc.T))

    result = temp.dot(sc.T)

    assert np.allclose(draws, result)
