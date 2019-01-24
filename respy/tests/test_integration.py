from pandas.util.testing import assert_frame_equal

import numpy as np
import pandas as pd
import pytest
import shutil
import copy
import glob

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.shared.shared_auxiliary import cholesky_to_coeffs
from respy.python.shared.shared_auxiliary import extract_cholesky
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.scripts.scripts_estimate import scripts_estimate
from respy.scripts.scripts_simulate import scripts_simulate
from respy.python.shared.shared_constants import IS_FORTRAN
from respy.pre_processing.data_processing import process_dataset
from respy.scripts.scripts_check import scripts_check
from respy.tests.codes.auxiliary import write_interpolation_grid
from respy.tests.codes.auxiliary import transform_to_logit
from respy.tests.codes.auxiliary import write_lagged_start
from respy.custom_exceptions import UserError
from respy.tests.codes.auxiliary import simulate_observed
from respy.tests.codes.random_model import generate_random_model
from respy.tests.codes.auxiliary import write_edu_start
from respy.tests.codes.auxiliary import write_types

from respy import RespyCls


class TestClass(object):
    """ This class groups together some tests.
    """

    def test_1(self):
        """Test if random model specifications can be simulated and processed."""
        params_spec, options_spec = generate_random_model()
        respy_obj = RespyCls(params_spec, options_spec)
        simulate_observed(respy_obj)
        process_dataset(respy_obj)

    def test_2(self):
        """ If there is no random variation in rewards then the number of draws to simulate the
        expected future value should have no effect.
        """
        params_spec, options_spec = generate_random_model(deterministic=True)

        # Initialize auxiliary objects
        base = None

        for _ in range(2):
            num_draws_emax = np.random.randint(1, 100)
            respy_obj = RespyCls(params_spec, options_spec)
            respy_obj.unlock()
            respy_obj.set_attr("num_draws_emax", num_draws_emax)
            respy_obj.lock()
            respy_obj = simulate_observed(respy_obj)
            periods_emax = respy_obj.get_attr("periods_emax")

            if base is None:
                base = periods_emax.copy()

            diff = np.max(
                abs(np.ma.masked_invalid(base) - np.ma.masked_invalid(periods_emax))
            )
            np.testing.assert_almost_equal(diff, 0.0)

    def test_3(self):
        """ Testing whether the a simulated dataset and the evaluation of the criterion function
        are the same for a tiny delta and a myopic agent.
        """
        constr = {"estimation": {"maxfun": 0}}
        params_spec, options_spec = generate_random_model(
            point_constr=constr, myopic=True)
        respy_obj = RespyCls(params_spec, options_spec)

        optim_paras, num_agents_sim, edu_spec = dist_class_attributes(
            respy_obj, "optim_paras", "num_agents_sim", "edu_spec"
        )

        write_types(optim_paras["type_shares"], num_agents_sim)
        write_edu_start(edu_spec, num_agents_sim)
        write_lagged_start(num_agents_sim)

        # Iterate over alternative discount rates.
        base_data, base_val = None, None

        for delta in [0.00, 0.000001]:

            respy_obj = RespyCls(params_spec, options_spec)

            respy_obj.unlock()

            respy_obj.attr["optim_paras"]["delta"] = np.array([delta])

            respy_obj.lock()

            simulate_observed(respy_obj)

            # This parts checks the equality of simulated dataset for the different versions of
            # the code.
            data_frame = pd.read_csv("data.respy.dat", delim_whitespace=True)

            if base_data is None:
                base_data = data_frame.copy()

            assert_frame_equal(base_data, data_frame)

            # This part checks the equality of an evaluation of the criterion function.
            _, crit_val = respy_obj.fit()

            if base_val is None:
                base_val = crit_val

            np.testing.assert_allclose(base_val, crit_val, rtol=1e-03, atol=1e-03)

    def test_4(self):
        """ Test the evaluation of the criterion function for random requests, not just at the
        true values.
        """
        # Constraints that ensure that two alternative initialization files can be used for the
        # same simulated data.

        num_agents = np.random.randint(5, 100)
        constr = {
            "simulation": {"agents": num_agents},
            "num_periods": np.random.randint(1, 4),
            "edu_spec": {"start": [7], "max": 15, "share": [1.0]},
            "estimation": {"maxfun": 0, "agents": num_agents}
        }

        # Simulate a dataset

        params_spec, options_spec = generate_random_model(point_constr=constr)
        respy_obj = RespyCls(params_spec, options_spec)
        simulate_observed(respy_obj)

        # Evaluate at different points, ensuring that the simulated dataset still fits.
        params_spec, options_spec = generate_random_model(point_constr=constr)
        respy_obj = RespyCls(params_spec, options_spec)
        respy_obj.fit()

    def test_5(self):
        """ Test the scripts.
        """
        # Constraints that ensure that two alternative initialization files can be used for the
        # same simulated data.
        for _ in range(10):
            num_agents = np.random.randint(5, 100)
            constr = {
                "simulation": {"agents": num_agents},
                "num_periods": np.random.randint(1, 4),
                "edu_spec": {"start": [7], "max": 15, "share": [1.0]},
                "estimation": {"maxfun": 0, "agents": num_agents}
            }
            # Simulate a dataset
            params_spec, options_spec = generate_random_model(point_constr=constr)
            respy_obj = RespyCls(params_spec, options_spec)
            simulate_observed(respy_obj)

            # Create output to process a baseline.
            respy_obj.unlock()
            respy_obj.set_attr("maxfun", 0)
            respy_obj.lock()

            respy_obj.fit()

            # Potentially evaluate at different points.
            params_spec, options_spec = generate_random_model(point_constr=constr)
            respy_obj = RespyCls(params_spec, options_spec)

            file_sim = "sim.respy.dat"

            single = np.random.choice([True, False])

            scripts_check("estimate", respy_obj)
            scripts_estimate(single, respy_obj)

    @pytest.mark.slow
    def test_6(self):
        """ Test short estimation tasks.
        """
        num_agents = np.random.randint(5, 100)
        constr = {
            "simulation": {"agents": num_agents},
            "num_periods": np.random.randint(1, 4),
            "estimation": {"maxfun": np.random.randint(0, 5), "agents": num_agents}
        }

        # Simulate a dataset

        params_spec, options_spec = generate_random_model(point_constr=constr)
        respy_obj = RespyCls(params_spec, options_spec)

        write_interpolation_grid(respy_obj)

        # Run estimation task.
        simulate_observed(respy_obj)
        base_x, base_val = respy_obj.fit()

        # We also check whether updating the class instance and a single evaluation of the
        # criterion function give the same result.
        respy_obj.update_optim_paras(base_x)
        respy_obj.attr["maxfun"] = 0

        alt_x, alt_val = respy_obj.fit()

        for arg in [(alt_val, base_val), (alt_x, base_x)]:
            np.testing.assert_almost_equal(arg[0], arg[1])

    def test_8(self):
        """ This test ensures that the constraints for the covariance matrix are properly handled.
        """

        params_spec, options_spec = generate_random_model()

        # Manual specification of update patterns.
        updates = dict()

        updates["valid_1"] = [
            False,
            True,
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            False,
        ]
        updates["valid_2"] = [False] * 10
        updates["valid_3"] = [True] * 10

        updates["invalid_1"] = [
            False,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            False,
        ]
        updates["invalid_2"] = [
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
        ]

        # We draw a random update and print it out to the initialization file.
        label = np.random.choice(list(updates.keys()))
        params_spec.loc['shocks', 'fixed'] = np.array(updates[label])

        if "invalid" in label:
            # This exception block makes sure that the UserError is in fact raised.
            try:
                RespyCls(params_spec, options_spec)
                raise AssertionError
            except UserError:
                pass
        else:
            RespyCls(params_spec, options_spec)

    def test_9(self):
        """ We ensure that the number of initial conditions does not matter for the evaluation of
        the criterion function if a weight of one is put on the first group.
        """
        num_agents = np.random.randint(5, 100)
        constr = {
            "simulation": {"agents": num_agents},
            "num_periods": np.random.randint(1, 4),
            "edu_spec": {"max": np.random.randint(15, 25, size=1).tolist()[0]},
            "estimation": {"maxfun": 0, "agents": num_agents},
            "interpolation": {"flag": False}
        }

        bound_constr = {}

        # Simulate a dataset

        params_spec, options_spec = generate_random_model(point_constr=constr)
        respy_obj = RespyCls(params_spec, options_spec)
        simulate_observed(respy_obj)


        base_val, edu_start_base = None, np.random.randint(1, 5, size=1).tolist()[0]

        # We need to ensure that the initial lagged activity always has the same distribution.
        edu_lagged_base = np.random.uniform(size=5).tolist()

        for num_edu_start in [1, np.random.choice([2, 3, 4]).tolist()]:

            # We always need to ensure that a weight of one is on the first level of initial
            # schooling.
            options_spec["edu_spec"]["share"] = [1.0] + [0.0] * (num_edu_start - 1)
            options_spec["edu_spec"]["lagged"] = edu_lagged_base[:num_edu_start]

            # We need to make sure that the baseline level of initial schooling is always
            # included. At the same time we cannot have any duplicates.
            edu_start = np.random.choice(
                range(1, 10), size=num_edu_start, replace=False
            ).tolist()
            if edu_start_base in edu_start:
                edu_start.remove(edu_start_base)
                edu_start.insert(0, edu_start_base)
            else:
                edu_start[0] = edu_start_base

            options_spec["edu_spec"]["start"] = edu_start

            respy_obj = RespyCls(params_spec, options_spec)
            simulate_observed(respy_obj)
            _, val = respy_obj.fit()
            if base_val is None:
                base_val = val

            np.testing.assert_almost_equal(base_val, val)

