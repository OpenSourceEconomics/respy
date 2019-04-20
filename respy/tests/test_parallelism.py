import numpy as np
import pytest

from respy import RespyCls
from respy.python.shared.shared_constants import IS_PARALLELISM_MPI
from respy.python.shared.shared_constants import IS_PARALLELISM_OMP
from respy.tests.codes.auxiliary import compare_est_log
from respy.tests.codes.auxiliary import simulate_observed
from respy.tests.codes.random_model import generate_random_model


@pytest.mark.skipif(
    not IS_PARALLELISM_MPI and not IS_PARALLELISM_OMP, reason="No PARALLELISM available"
)
class TestClass(object):
    """This class groups together some tests."""

    def test_1(self):
        """Ensure that it makes no difference whether the
        criterion function is evaluated in parallel or not.
        """
        # Generate random initialization file
        constr = {
            "program": {"version": "fortran"},
            "estimation": {"maxfun": np.random.randint(0, 50)},
        }

        params_spec, options_spec = generate_random_model(point_constr=constr)

        # If delta is a not fixed, we need to ensure a bound-constraint optimizer.
        # However, this is not the standard flag_estimation as the number of function evaluation
        # is possibly much larger to detect and differences in the updates of the optimizer
        # steps depending on the implementation.
        if params_spec.loc[("delta", "delta"), "fixed"] == False:
            options_spec["estimation"]["optimizer"] = "FORT-BOBYQA"

        base = None
        for is_parallel in [True, False]:
            options_spec["program"]["threads"] = 1
            options_spec["program"]["procs"] = 1

            if is_parallel:
                if IS_PARALLELISM_OMP:
                    options_spec["program"]["threads"] = np.random.randint(2, 5)
                if IS_PARALLELISM_MPI:
                    options_spec["program"]["procs"] = np.random.randint(2, 5)

            respy_obj = RespyCls(params_spec, options_spec)
            respy_obj = simulate_observed(respy_obj)
            _, crit_val = respy_obj.fit()

            if base is None:
                base = crit_val
            np.testing.assert_equal(base, crit_val)

    def test_2(self):
        """ This test ensures that the record files are identical.
        """
        # Generate random initialization file. The number of periods is higher than usual as only
        # FORTRAN implementations are used to solve the random request. This ensures that also
        # some cases of interpolation are explored.
        constr = {
            "program": {"version": "fortran"},
            "num_periods": np.random.randint(3, 10),
            "estimation": {"maxfun": 0},
        }

        params_spec, options_spec = generate_random_model(point_constr=constr)

        base_sol_log, base_est_info_log = None, None
        base_est_log = None

        for is_parallel in [False, True]:

            options_spec["program"]["threads"] = 1
            options_spec["program"]["procs"] = 1

            if is_parallel:
                if IS_PARALLELISM_OMP:
                    options_spec["program"]["threads"] = np.random.randint(2, 5)
                if IS_PARALLELISM_MPI:
                    options_spec["program"]["procs"] = np.random.randint(2, 5)

            respy_obj = RespyCls(params_spec, options_spec)

            file_sim = respy_obj.get_attr("file_sim")

            simulate_observed(respy_obj)

            respy_obj.fit()

            # Check for identical records
            fname = file_sim + ".respy.sol"

            if base_sol_log is None:
                base_sol_log = open(fname, "r").read()

            assert open(fname, "r").read() == base_sol_log

            if base_est_info_log is None:
                base_est_info_log = open("est.respy.info", "r").read()
            assert open("est.respy.info", "r").read() == base_est_info_log

            if base_est_log is None:
                base_est_log = open("est.respy.log", "r").readlines()
            compare_est_log(base_est_log)
