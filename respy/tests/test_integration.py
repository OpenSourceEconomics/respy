import copy

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal

from respy import RespyCls
from respy.python.shared.shared_auxiliary import cholesky_to_coeffs
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import extract_cholesky
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.tests.codes.auxiliary import simulate_observed
from respy.tests.codes.auxiliary import write_interpolation_grid
from respy.tests.codes.random_model import generate_random_model


class TestClass(object):
    """ This class groups together some tests.
    """

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

    @pytest.mark.slow
    def test_6(self):
        """Test short estimation tasks."""
        num_agents = np.random.randint(5, 100)
        constr = {
            "simulation": {"agents": num_agents},
            "num_periods": np.random.randint(1, 4),
            "estimation": {"maxfun": np.random.randint(0, 5), "agents": num_agents},
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

    def test_8(self):
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

        base_val, edu_start_base = (None, np.random.randint(1, 5, size=1).tolist()[0])

        # We need to ensure that the initial lagged activity always has the same
        # distribution.
        edu_lagged_base = np.random.uniform(size=5).tolist()

        for num_edu_start in [1, np.random.choice([2, 3, 4]).tolist()]:

            # We always need to ensure that a weight of one is on the first level of
            # initial schooling.
            options_spec["edu_spec"]["share"] = [1.0] + [0.0] * (num_edu_start - 1)
            options_spec["edu_spec"]["lagged"] = edu_lagged_base[:num_edu_start]

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

    def test_9(self):
        """ This test ensures that the order of the initial schooling level specified in
        the initialization files does not matter for the simulation of a dataset and
        subsequent evaluation of the criterion function.

        Warning
        -------
        This test fails if types have the identical intercept as no unique ordering is
        determined than.

        """
        point_constr = {
            "estimation": {"maxfun": 0},
            # We cannot allow for interpolation as the order of states within each
            # period changes and thus the prediction model is altered even if the same
            # state identifier is used.
            "interpolation": {"flag": False},
        }

        params_spec, options_spec = generate_random_model(point_constr=point_constr)

        respy_obj = RespyCls(params_spec, options_spec)

        edu_baseline_spec, num_types, num_paras, optim_paras = dist_class_attributes(
            respy_obj, "edu_spec", "num_types", "num_paras", "optim_paras"
        )

        # We want to randomly shuffle the list of initial schooling but need to maintain
        # the order of the shares.
        edu_shuffled_start = np.random.permutation(edu_baseline_spec["start"]).tolist()

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

        list_ = [optim_paras["type_shifts"][i, :].tolist() for i in types_order]
        optim_paras_shuffled["type_shifts"] = np.array(list_)

        list_ = [type_shares[i] for i in types_order]
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
                data_frame = pd.read_csv("data.respy.dat", delim_whitespace=True)

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
