import numpy as np
import pytest

from respy import RespyCls
from respy.pre_processing.model_processing import _options_spec_from_attributes
from respy.pre_processing.model_processing import _params_spec_from_attributes
from respy.python.shared.shared_constants import TOL
from respy.tests.codes.auxiliary import simulate_observed


params = [
    0,
    1,
    2,
    pytest.param(3, marks=pytest.mark.xfail),
    pytest.param(4, marks=pytest.mark.xfail),
    pytest.param(5, marks=pytest.mark.xfail),
    6,
    pytest.param(7, marks=pytest.mark.xfail),
    pytest.param(8, marks=pytest.mark.xfail),
    pytest.param(9, marks=pytest.mark.xfail),
]


@pytest.mark.parametrize("index", params)
def test_single_regression(regression_vault, index):
    """Run a single regression test."""
    attr, crit_val = regression_vault[index]

    params_spec = _params_spec_from_attributes(attr)
    options_spec = _options_spec_from_attributes(attr)
    respy_obj = RespyCls(params_spec, options_spec)

    simulate_observed(respy_obj)

    est_val = respy_obj.fit()[1]

    assert np.isclose(est_val, crit_val, rtol=TOL, atol=TOL)
