import numpy as np
import pytest

from respy.python.interface import minimal_estimation_interface
from respy.python.shared.shared_constants import TOL
from respy.tests.codes.auxiliary import minimal_simulate_observed


params = [
    0,
    1,
    pytest.param(2, marks=pytest.mark.xfail),
    pytest.param(3, marks=pytest.mark.xfail),
    pytest.param(4, marks=pytest.mark.xfail),
    pytest.param(5, marks=pytest.mark.xfail),
    6,
    pytest.param(7, marks=pytest.mark.xfail),
    pytest.param(8, marks=pytest.mark.xfail),
    pytest.param(9, marks=pytest.mark.xfail),
    10,
    11,
    12,
    pytest.param(13, marks=pytest.mark.xfail),
    pytest.param(14, marks=pytest.mark.xfail),
    15,
    pytest.param(16, marks=pytest.mark.xfail),
    17,
    18,
    pytest.param(19, marks=pytest.mark.xfail),
]


@pytest.mark.parametrize("index", params)
def test_single_regression(regression_vault, index):
    """Run a single regression test."""
    attr, crit_val = regression_vault[index]

    df = minimal_simulate_observed(attr)

    x, est_val = minimal_estimation_interface(attr, df)

    assert np.isclose(est_val, crit_val, rtol=TOL, atol=TOL)
