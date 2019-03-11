from pandas.util.testing import assert_frame_equal

import copy
import numpy as np
import pandas as pd
import pytest
import random
import shutil

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.shared.shared_auxiliary import cholesky_to_coeffs
from respy.python.shared.shared_auxiliary import get_valid_bounds
from respy.python.shared.shared_auxiliary import extract_cholesky
from respy.pre_processing.model_processing import write_init_file
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.scripts.scripts_estimate import scripts_estimate
from respy.scripts.scripts_simulate import scripts_simulate
from respy.python.shared.shared_constants import IS_FORTRAN
from respy.scripts.scripts_update import scripts_update
from respy.scripts.scripts_modify import scripts_modify
from respy.pre_processing.data_processing import process_dataset
from respy.scripts.scripts_check import scripts_check
from respy.tests.codes.auxiliary import write_interpolation_grid
from respy.tests.codes.random_init import generate_random_dict
from respy.tests.codes.auxiliary import transform_to_logit
from respy.tests.codes.auxiliary import write_lagged_start
from respy.custom_exceptions import UserError
from respy.tests.codes.auxiliary import simulate_observed
from respy.tests.codes.random_init import generate_init
from respy.tests.codes.auxiliary import write_edu_start
from respy.tests.codes.auxiliary import compare_init
from respy.tests.codes.auxiliary import write_types

from respy import RespyCls


class TestClass(object):
    """ This class groups together some tests.
    """

    def test_1(self):
        """ Testing whether random model specifications can be simulated and processed.
        """
        # Generate random initialization file
        generate_init()

        respy_obj = RespyCls("test.respy.ini")

        simulate_observed(respy_obj)

        process_dataset(respy_obj)

    def test_2(self):
        """ If there is no random variation in rewards then the number of draws to simulate the
        expected future value should have no effect.
        """
        # Generate constraints
        constr = dict()
        constr["flag_deterministic"] = True

        # Generate random initialization file
        generate_init(constr)

        # Initialize auxiliary objects
        base = None

        for _ in range(2):

            # Draw a random number of draws for
            # expected future value calculations.
            num_draws_emax = np.random.randint(1, 100)

            # Perform toolbox actions
            respy_obj = RespyCls("test.respy.ini")

            respy_obj.unlock()

            respy_obj.set_attr("num_draws_emax", num_draws_emax)

            respy_obj.lock()

            respy_obj = simulate_observed(respy_obj)

            # Distribute class attributes
            periods_emax = respy_obj.get_attr("periods_emax")

            if base is None:
                base = periods_emax.copy()

            # Statistic
            diff = np.max(
                abs(
                    np.ma.masked_invalid(base)
                    - np.ma.masked_invalid(periods_emax)
                )
            )

            # Checks
            np.testing.assert_almost_equal(diff, 0.0)

    def test_3(self):
        """ Testing whether the a simulated dataset and the evaluation of the criterion function
        are the same for a tiny delta and a myopic agent.
        """
        # Generate random initialization dictionary
        constr = dict()
        constr["maxfun"] = 0
        constr["flag_myopic"] = True

        generate_init(constr)
        respy_obj = RespyCls("test.respy.ini")

        optim_paras, num_agents_sim, edu_spec = dist_class_attributes(
            respy_obj, "optim_paras", "num_agents_sim", "edu_spec"
        )

        write_types(optim_paras["type_shares"], num_agents_sim)
        write_edu_start(edu_spec, num_agents_sim)
        write_lagged_start(num_agents_sim)

        # Iterate over alternative discount rates.
        base_data, base_val = None, None

        for delta in [0.00, 0.000001]:

            respy_obj = RespyCls("test.respy.ini")

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
        constr = dict()
        constr["agents"] = np.random.randint(1, 100)
        constr["periods"] = np.random.randint(1, 4)
        constr["edu"] = (7, 15)
        constr["maxfun"] = 0

        # Simulate a dataset
        generate_init(constr)
        respy_obj = RespyCls("test.respy.ini")
        simulate_observed(respy_obj)

        # Evaluate at different points, ensuring that the simulated dataset still fits.
        generate_init(constr)

        respy_obj = RespyCls("test.respy.ini")
        respy_obj.fit()

    def test_5(self):
        """ Test the scripts.
        """
        # Constraints that ensure that two alternative initialization files can be used
        # for the same simulated data.
        for _ in range(10):
            constr = dict()
            constr["periods"] = np.random.randint(1, 4)
            constr["agents"] = np.random.randint(5, 100)
            constr["flag_estimation"] = True
            constr["edu"] = (7, 15)

            # Simulate a dataset
            generate_init(constr)
            respy_obj = RespyCls("test.respy.ini")
            simulate_observed(respy_obj)

            # Create output to process a baseline.
            respy_obj.unlock()
            respy_obj.set_attr("maxfun", 0)
            respy_obj.lock()

            respy_obj.fit()

            # Potentially evaluate at different points.
            generate_init(constr)

            init_file = "test.respy.ini"
            file_sim = "sim.respy.dat"

            single = np.random.choice([True, False])

            scripts_check("estimate", init_file)
            scripts_estimate(single, init_file)
            scripts_update(init_file)

            action = np.random.choice(["fix", "free", "bounds", "values"])

            # The set of identifiers is a little complicated as we only allow sampling
            # of the diagonal terms of the covariance matrix. Otherwise, we sometimes
            # run into the problem of very ill conditioned matrices resulting in a
            # failed Cholesky decomposition.
            valid_identifiers = list(range(43))

            respy_obj = RespyCls("test.respy.ini")
            optim_paras, num_paras = dist_class_attributes(
                respy_obj, "optim_paras", "num_paras"
            )

            # Now we need to handle some different case distinctions.
            if action in ["bounds"]:
                identifiers = np.random.choice(valid_identifiers, size=2)[0:1]
                value = get_optim_paras(optim_paras, num_paras, "all", True)[
                    identifiers
                ][0]

                if identifiers in [0]:
                    which = "delta"
                else:
                    which = "coeff"

                bounds = get_valid_bounds(which, value)
                values = None

            elif action in ["fix", "free"]:
                num_draws = np.random.randint(1, 43)
                identifiers = np.random.choice(
                    valid_identifiers, num_draws, replace=False
                )
                bounds, values = None, None
            else:
                # This is restrictive in the sense that the original value will be used
                # for the replacement. However, otherwise the handling of the bounds
                # requires too much effort at this point.
                num_draws = np.random.randint(1, 43)
                identifiers = np.random.choice(
                    valid_identifiers, num_draws, replace=False
                )
                values = get_optim_paras(optim_paras, num_paras, "all", True)[
                    identifiers
                ]
                bounds = None

            scripts_simulate(init_file, file_sim)
            scripts_modify(identifiers, action, init_file, values, bounds)

    @pytest.mark.slow
    def test_6(self):
        """ Test short estimation tasks.
        """
        constr = dict()
        constr["flag_estimation"] = True

        generate_init(constr)

        write_interpolation_grid("test.respy.ini")

        # Run estimation task.
        respy_obj = RespyCls("test.respy.ini")
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
        """ We test whether a restart does result in the exact function evaluation.
        Additionally, we change the status of parameters at random.
        """
        constr = dict()
        constr["flag_estimation"] = True
        generate_init(constr)

        respy_obj = RespyCls("test.respy.ini")
        respy_obj = simulate_observed(respy_obj)
        _, base_val = respy_obj.fit()

        scripts_update("test.respy.ini")
        respy_obj = RespyCls("test.respy.ini")

        respy_obj.unlock()
        respy_obj.set_attr("maxfun", 0)
        respy_obj.lock()

        action = np.random.choice(["fix", "free"])
        num_draws = np.random.randint(1, 18)
        identifiers = np.random.choice(range(18), num_draws, replace=False)
        scripts_modify(identifiers, action, "test.respy.ini")

        _, update_val = respy_obj.fit()

        np.testing.assert_almost_equal(update_val, base_val)

    def test_8(self):
        """ This test ensures that the constraints for the covariance matrix are
        properly handled.
        """

        init_dict = generate_random_dict()

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
        init_dict["SHOCKS"]["fixed"] = updates[label]
        write_init_file(init_dict)

        if "invalid" in label:
            # This exception block makes sure that the UserError is in fact raised.
            try:
                RespyCls("test.respy.ini")
                raise AssertionError
            except UserError:
                pass
        else:
            RespyCls("test.respy.ini")

    def test_9(self):
        """ We ensure that the number of initial conditions does not matter for the
        evaluation of the criterion function if a weight of one is put on the first
        group.
        """
        constr = dict()
        constr["flag_estimation"] = True

        # The interpolation equation is affected by the number of types regardless of
        # the weights.
        constr["flag_interpolation"] = False

        init_dict = generate_init(constr)

        base_val, edu_start_base = (
            None,
            np.random.randint(1, 5, size=1).tolist()[0],
        )

        # We need to make sure that the maximum level of education is the same across
        # iterations, but also larger than any of the initial starting levels.
        init_dict["EDUCATION"]["max"] = np.random.randint(
            15, 25, size=1
        ).tolist()[0]

        # We need to ensure that the initial lagged activity always has the same
        # distribution.
        edu_lagged_base = np.random.uniform(size=5).tolist()

        for num_edu_start in [1, np.random.choice([2, 3, 4]).tolist()]:

            # We always need to ensure that a weight of one is on the first level of
            # initial schooling.
            init_dict["EDUCATION"]["share"] = [1.0] + [0.0] * (
                num_edu_start - 1
            )
            init_dict["EDUCATION"]["lagged"] = edu_lagged_base[:num_edu_start]

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

            init_dict["EDUCATION"]["start"] = edu_start

            write_init_file(init_dict)

            respy_obj = RespyCls("test.respy.ini")
            simulate_observed(respy_obj)
            _, val = respy_obj.fit()
            if base_val is None:
                base_val = val

            np.testing.assert_almost_equal(base_val, val)

    @pytest.mark.skipif(not IS_FORTRAN, reason="No FORTRAN available")
    def test_11(self):
        """ This step ensures that the printing of the initialization file is done properly. We
        compare the content of the files line by line, but drop any spaces.
        """
        for fname in TEST_RESOURCES_DIR.glob("*.ini"):
            respy_obj = RespyCls(fname)
            respy_obj.write_out("test.respy.ini")
            np.testing.assert_equal(
                compare_init(fname, "test.respy.ini"), True
            )

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
            k=5,
        ),
    )
    def test_12(self, fname, result):
        """ This test just locks in the evaluation of the criterion function for the
        original Keane & Wolpin data. We create an additional initialization files that
        include numerous types and initial conditions.

        """
        # This ensures that the experience effect is taken care of properly.
        open(".restud.respy.scratch", "w").close()

        # Evaluate criterion function at true values.
        respy_obj = RespyCls(TEST_RESOURCES_DIR / fname)

        respy_obj.unlock()
        respy_obj.set_attr("maxfun", 0)
        respy_obj.lock()

        simulate_observed(respy_obj, is_missings=False)

        _, val = respy_obj.fit()
        np.testing.assert_allclose(val, result)

    def test_13(self):
        """ We ensure that the number of types does not matter for the evaluation of the
        criterion function if a weight of one is put on the first group.
        """

        constr = dict()
        constr["flag_estimation"] = True

        # The interpolation equation is affected by the number of types regardless of
        # the weights.
        constr["flag_interpolation"] = False

        init_dict = generate_init(constr)

        base_val = None

        for num_types in [1, np.random.choice([2, 3, 4]).tolist()]:

            # We construct a set of coefficients that matches the type shares.
            coeffs = transform_to_logit([1.0] + [0.00000001] * (num_types - 1))

            # We always need to ensure that a weight of one is on the first type. We
            # need to fix the weight in this case to not change during an estimation as
            # well.
            shifts = np.random.uniform(-0.05, 0.05, size=(num_types - 1) * 4)
            init_dict["TYPE SHIFTS"]["coeffs"] = shifts
            init_dict["TYPE SHIFTS"]["fixed"] = [False] * (num_types * 4)
            init_dict["TYPE SHIFTS"]["bounds"] = [[None, None]] * (
                num_types * 4
            )

            init_dict["TYPE SHARES"]["coeffs"] = coeffs[2:]
            init_dict["TYPE SHARES"]["fixed"] = [True, True] * (num_types - 1)
            init_dict["TYPE SHARES"]["bounds"] = [
                [None, None],
                [None, None],
            ] * (num_types - 1)

            write_init_file(init_dict)

            respy_obj = RespyCls("test.respy.ini")
            simulate_observed(respy_obj)
            _, val = respy_obj.fit()
            if base_val is None:
                base_val = val

            np.testing.assert_almost_equal(base_val, val)

    def test_14(self):
        """ Testing the modification routine.
        """

        generate_init()

        respy_obj = RespyCls("test.respy.ini")
        num_paras, optim_paras = dist_class_attributes(
            respy_obj, "num_paras", "optim_paras"
        )

        # We need to switch perspective where the initialization file is the flattened
        # covariance matrix.
        x_econ = get_optim_paras(optim_paras, num_paras, "all", True)
        x_econ[43:53] = cholesky_to_coeffs(optim_paras["shocks_cholesky"])

        # We now draw a random set of points where we replace the value with their
        # actual value. Thus, there should be no differences in the initialization files
        # afterwards.
        shutil.copy("test.respy.ini", "baseline.respy.ini")

        num_subset = np.random.randint(low=0, high=num_paras)
        identifiers = np.random.choice(
            range(num_paras), size=num_subset, replace=False
        )

        scripts_modify(
            identifiers, "values", "test.respy.ini", values=x_econ[identifiers]
        )
        np.testing.assert_equal(
            compare_init("baseline.respy.ini", "test.respy.ini"), True
        )

    def test_15(self):
        """ This test ensures that the order of the initial schooling level specified in
        the initialization files does not matter for the simulation of a dataset and
        subsequent evaluation of the criterion function.

        WARNING: This test fails if types have the identical intercept as no unique
        ordering is determined than.
        """

        constr = dict()
        constr["maxfun"] = 0

        # We cannot allow for interpolation as the order of states within each period
        # changes and thus the prediction model is altered even if the same state
        # identifier is used.
        constr["flag_interpolation"] = False
        generate_init(constr)

        respy_obj = RespyCls("test.respy.ini")

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
