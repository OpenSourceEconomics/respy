from pandas.util.testing import assert_frame_equal

import copy
import numpy as np
import pandas as pd
import pytest
import random

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.shared.shared_auxiliary import cholesky_to_coeffs
from respy.python.shared.shared_auxiliary import extract_cholesky
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.scripts.scripts_estimate import scripts_estimate
from respy.python.shared.shared_constants import IS_FORTRAN
from respy.pre_processing.data_processing import process_dataset
from respy.scripts.scripts_check import scripts_check
from respy.tests.codes.auxiliary import write_interpolation_grid
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
                abs(
                    np.ma.masked_invalid(base)
                    - np.ma.masked_invalid(periods_emax)
                )
            )
            np.testing.assert_almost_equal(diff, 0.0)

    def test_3(self):
        """ Testing whether the a simulated dataset and the evaluation of the criterion function
        are the same for a tiny delta and a myopic agent.
        """
        constr = {"estimation": {"maxfun": 0}}
        params_spec, options_spec = generate_random_model(
            point_constr=constr, myopic=True
        )
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

            # This parts checks the equality of simulated dataset for the different
            # versions of the code.
            data_frame = pd.read_csv("data.respy.dat", delim_whitespace=True)

            if base_data is None:
                base_data = data_frame.copy()

            assert_frame_equal(base_data, data_frame)

            # This part checks the equality of an evaluation of the criterion function.
            _, crit_val = respy_obj.fit()

            if base_val is None:
                base_val = crit_val

            np.testing.assert_allclose(
                base_val, crit_val, rtol=1e-03, atol=1e-03
            )

    def test_4(self):
        """ Test the evaluation of the criterion function for random requests, not just
        at the true values.
        """
        # Constraints that ensure that two alternative initialization files can be used
        # for the same simulated data.

        num_agents = np.random.randint(5, 100)
        constr = {
            "simulation": {"agents": num_agents},
            "num_periods": np.random.randint(1, 4),
            "edu_spec": {"start": [7], "max": 15, "share": [1.0]},
            "estimation": {"maxfun": 0, "agents": num_agents},
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
        # Constraints that ensure that two alternative initialization files can be used
        # for the same simulated data.
        for _ in range(10):
            num_agents = np.random.randint(5, 100)
            constr = {
                "simulation": {"agents": num_agents},
                "num_periods": np.random.randint(1, 4),
                "edu_spec": {"start": [7], "max": 15, "share": [1.0]},
                "estimation": {"maxfun": 0, "agents": num_agents},
            }
            # Simulate a dataset
            params_spec, options_spec = generate_random_model(
                point_constr=constr
            )
            respy_obj = RespyCls(params_spec, options_spec)
            simulate_observed(respy_obj)

            # Create output to process a baseline.
            respy_obj.unlock()
            respy_obj.set_attr("maxfun", 0)
            respy_obj.lock()

            respy_obj.fit()

            # Potentially evaluate at different points.
            params_spec, options_spec = generate_random_model(
                point_constr=constr
            )
            respy_obj = RespyCls(params_spec, options_spec)

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
            "estimation": {
                "maxfun": np.random.randint(0, 5),
                "agents": num_agents,
            },
        }

        # Simulate a dataset

        params_spec, options_spec = generate_random_model(point_constr=constr)
        respy_obj = RespyCls(params_spec, options_spec)

        write_interpolation_grid(respy_obj)

        # Run estimation task.
        simulate_observed(respy_obj)
        base_x, base_val = respy_obj.fit()

        # We also check whether updating the class instance and a single evaluation of
        # the criterion function give the same result.
        respy_obj.update_optim_paras(base_x)
        respy_obj.attr["maxfun"] = 0

        alt_x, alt_val = respy_obj.fit()

        for arg in [(alt_val, base_val), (alt_x, base_x)]:
            np.testing.assert_almost_equal(arg[0], arg[1])

    def test_7(self):
        """ This test ensures that the constraints for the covariance matrix are
        properly handled.
        """

        params_spec, options_spec = generate_random_model(deterministic=True)

        # Manual specification of update patterns.
        updates = dict()

        # off-diagonals fixed
        updates["valid_1"] = [
            False,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
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
        params_spec.loc["shocks", "fixed"] = np.array(updates[label])

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
        """ We ensure that the number of initial conditions does not matter for the
        evaluation of the criterion function if a weight of one is put on the first
        group.
        """
        num_agents = np.random.randint(5, 100)
        constr = {
            "simulation": {"agents": num_agents},
            "num_periods": np.random.randint(1, 4),
            "edu_spec": {"max": np.random.randint(15, 25, size=1).tolist()[0]},
            "estimation": {"maxfun": 0, "agents": num_agents},
            "interpolation": {"flag": False},
        }

        params_spec, options_spec = generate_random_model(point_constr=constr)
        respy_obj = RespyCls(params_spec, options_spec)
        simulate_observed(respy_obj)

        base_val, edu_start_base = (
            None,
            np.random.randint(1, 5, size=1).tolist()[0],
        )

        # We need to ensure that the initial lagged activity always has the same
        # distribution.
        edu_lagged_base = np.random.uniform(size=5).tolist()

        for num_edu_start in [1, np.random.choice([2, 3, 4]).tolist()]:

            # We always need to ensure that a weight of one is on the first level of
            # initial schooling.
            options_spec["edu_spec"]["share"] = [1.0] + [0.0] * (
                num_edu_start - 1
            )
            options_spec["edu_spec"]["lagged"] = edu_lagged_base[
                :num_edu_start
            ]

            # We need to make sure that the baseline level of initial schooling is
            # always included. At the same time we cannot have any duplicates.
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

    @pytest.mark.skipif(not IS_FORTRAN, reason="No FORTRAN available")
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fname, result",
        random.sample(
            [
                ("kw_data_one.ini", 10.45950941513551),
                ("kw_data_two.ini", 45.04552402391903),
                ("kw_data_three.ini", 74.28253652773714),
                ("kw_data_one_types.ini", 9.098738585839529),
                ("kw_data_one_initial.ini", 7.965979149372883),
            ],
            k=1,
        ),
    )
    def test_12(self, fname, result):
        """ This test just locks in the evaluation of the criterion function for the
        original Keane & Wolpin data. We create an additional initialization files that
        include numerous types and initial conditions.

        """
        # This ensures that the experience effect is taken care of properly.
        open(".restud.respy.scratch", "w").close()

        base_path = TEST_RESOURCES_DIR / fname

        # Evaluate criterion function at true values.
        respy_obj = RespyCls(
            str(base_path.with_suffix(".csv")),
            str(base_path.with_suffix(".json")),
        )

        respy_obj.unlock()
        respy_obj.set_attr("maxfun", 0)
        respy_obj.lock()

        simulate_observed(respy_obj, is_missings=False)

        _, val = respy_obj.fit()

        np.testing.assert_allclose(val, result)

    def test_15(self):
        """ This test ensures that the order of the initial schooling level specified in
        the initialization files does not matter for the simulation of a dataset and
        subsequent evaluation of the criterion function.

        WARNING: This test fails if types have the identical intercept as no unique
        ordering is determined than.
        """

        point_constr = {
            "estimation": {"maxfun": 0},
            # We cannot allow for interpolation as the order of states within each
            # period changes and thus the prediction model is altered even if the same
            # state identifier is used.
            "interpolation": {"flag": False},
        }

        params_spec, options_spec = generate_random_model(
            point_constr=point_constr
        )

        respy_obj = RespyCls(params_spec, options_spec)

        edu_baseline_spec, num_types, num_paras, optim_paras = dist_class_attributes(
            respy_obj, "edu_spec", "num_types", "num_paras", "optim_paras"
        )

        # We want to randomly shuffle the list of initial schooling but need to maintain
        # the order of the shares.
        edu_shuffled_start = np.random.permutation(
            edu_baseline_spec["start"]
        ).tolist()

        edu_shuffled_share, edu_shuffled_lagged = [], []
        for start in edu_shuffled_start:
            idx = edu_baseline_spec["start"].index(start)
            edu_shuffled_lagged += [edu_baseline_spec["lagged"][idx]]
            edu_shuffled_share += [edu_baseline_spec["share"][idx]]

        edu_shuffled_spec = copy.deepcopy(edu_baseline_spec)
        edu_shuffled_spec["lagged"] = edu_shuffled_lagged
        edu_shuffled_spec["start"] = edu_shuffled_start
        edu_shuffled_spec["share"] = edu_shuffled_share

        # We are only looking at a single evaluation as otherwise the reordering affects
        # the optimizer that is trying better parameter values one-by-one. The
        # reordering might also violate the bounds.
        for i in range(53, num_paras):
            optim_paras["paras_bounds"][i] = [None, None]
            optim_paras["paras_fixed"][i] = False

        # We need to ensure that the baseline type is still in the first position.
        types_order = [0] + np.random.permutation(range(1, num_types)).tolist()

        type_shares = []
        for i in range(num_types):
            lower, upper = i * 2, (i + 1) * 2
            type_shares += [optim_paras["type_shares"][lower:upper].tolist()]

        optim_paras_baseline = copy.deepcopy(optim_paras)
        optim_paras_shuffled = copy.deepcopy(optim_paras)

        list_ = list(
            optim_paras["type_shifts"][i, :].tolist() for i in types_order
        )
        optim_paras_shuffled["type_shifts"] = np.array(list_)

        list_ = list(type_shares[i] for i in types_order)
        optim_paras_shuffled["type_shares"] = np.array(list_).flatten()

        base_data, base_val = None, None

        k = 0

        for optim_paras in [optim_paras_baseline, optim_paras_shuffled]:
            for edu_spec in [edu_baseline_spec, edu_shuffled_spec]:

                respy_obj.unlock()
                respy_obj.set_attr("edu_spec", edu_spec)
                respy_obj.lock()

                # There is some more work to do to update the coefficients as we
                # distinguish between the economic and optimization version of the
                # parameters.
                x = get_optim_paras(optim_paras, num_paras, "all", True)
                shocks_cholesky, _ = extract_cholesky(x)
                shocks_coeffs = cholesky_to_coeffs(shocks_cholesky)
                x[43:53] = shocks_coeffs
                respy_obj.update_optim_paras(x)

                respy_obj.reset()

                simulate_observed(respy_obj)

                # This part checks the equality of simulated dataset.
                data_frame = pd.read_csv(
                    "data.respy.dat", delim_whitespace=True
                )

                if base_data is None:
                    base_data = data_frame.copy()

                assert_frame_equal(base_data, data_frame)

                # This part checks the equality of a single function evaluation.
                _, val = respy_obj.fit()
                if base_val is None:
                    base_val = val
                np.testing.assert_almost_equal(base_val, val)

                respy_obj.reset()
                k += 1
