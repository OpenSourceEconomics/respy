import numpy as np
import pytest

from respy import RespyCls
from respy.pre_processing.model_processing import _options_spec_from_attributes
from respy.pre_processing.model_processing import _params_spec_from_attributes
from respy.python.shared.shared_constants import IS_FORTRAN
from respy.python.shared.shared_constants import IS_PARALLELISM_MPI
from respy.python.shared.shared_constants import IS_PARALLELISM_OMP
from respy.python.shared.shared_constants import TOL
from respy.tests.codes.auxiliary import simulate_observed


@pytest.mark.parametrize("index", range(10))
def test_single_regression(regression_vault, index):
    """This function checks a single test from the dictionary."""
    # Distribute test information.
    attr, crit_val = regression_vault[index]

    if not IS_PARALLELISM_OMP or not IS_FORTRAN:
        attr["num_threads"] = 1

    if not IS_PARALLELISM_MPI or not IS_FORTRAN:
        attr["num_procs"] = 1

    if not IS_FORTRAN:
        attr["version"] = "python"

    params_spec = _params_spec_from_attributes(attr)
    options_spec = _options_spec_from_attributes(attr)
    respy_obj = RespyCls(params_spec, options_spec)

    simulate_observed(respy_obj)

    est_val = respy_obj.fit()[1]

    assert np.isclose(est_val, crit_val, rtol=TOL, atol=TOL)
