import pytest
import numpy as np
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import distribute_parameters
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.tests.codes.random_model import generate_random_model
from respy.pre_processing.model_processing import (
    _read_options_spec,
    _read_params_spec,
)
from respy import RespyCls
from pandas.testing import assert_series_equal


class TestClass(object):
    """ This class groups together some tests.
    """

    def test_1(self):
        """ Testing whether back-and-forth transformation have no effect.
        """
        for i in range(10):
            num_types = np.random.randint(1, 5)
            num_paras = 53 + (num_types - 1) * 6

            # Create random parameter vector
            base = np.random.uniform(size=num_paras)

            x = base.copy()

            # Apply numerous transformations
            for j in range(10):
                optim_paras = distribute_parameters(x, is_debug=True)
                args = (optim_paras, num_paras, "all", True)
                x = get_optim_paras(*args)

            np.testing.assert_allclose(base, x)

    def test_2(self):
        """ Testing whether the back and forth transformation works.
        """
        for _ in range(100):
            params_spec, options_spec = generate_random_model()
            # Process request and write out again.
            respy_obj = RespyCls(params_spec, options_spec)
            respy_obj.write_out("alt.respy")

            new_params_spec = _read_params_spec("alt.respy.csv")
            new_options_spec = _read_options_spec("alt.respy.json")

            assert options_spec == new_options_spec

            for col in params_spec.columns:
                assert_series_equal(params_spec[col], new_params_spec[col])

    def test_3(self):
        """ Testing some of the relationships in the simulated dataset.
        """
        is_deterministic = np.random.choice([True, False])
        is_myopic = np.random.choice([True, False])

        max_draws = np.random.randint(5, 200)
        bound_constr = {"max_draws": max_draws, "max_agents": max_draws}

        params_spec, options_spec = generate_random_model(
            bound_constr=bound_constr,
            deterministic=is_deterministic,
            myopic=is_myopic,
        )

        respy_obj = RespyCls(params_spec, options_spec)
        _, df = respy_obj.simulate()

        optim_paras, num_types, edu_spec, num_periods = dist_class_attributes(
            respy_obj, "optim_paras", "num_types", "edu_spec", "num_periods"
        )

        # We can back out the wage information from other information provided in the
        # simulated dataset.
        for choice in [1, 2]:
            cond = df["Choice"] == choice
            label_sys = "Systematic_Reward_{}".format(choice)
            label_sho = "Shock_Reward_{}".format(choice)
            label_gen = "General_Reward_{}".format(choice)
            label_com = "Common_Reward"
            df["Ex_Post_Reward"] = (
                df[label_sys] - df[label_gen] - df[label_com]
            ) * df[label_sho]

            col_1 = df["Ex_Post_Reward"].loc[:, cond]
            col_2 = df["Wage"].loc[:, cond]
            np.testing.assert_array_almost_equal(col_1, col_2)

        # In the myopic case, the total reward should the equal to the ex post rewards.
        if is_myopic:
            # The shock only affects the skill-function and not the other components
            # determining the overall reward.
            for choice in [1, 2]:
                cond = df["Choice"] == choice

                label = "Ex_Post_Reward_{}".format(choice)
                label_gen = "General_Reward_{}".format(choice)
                label_com = "Common_Reward"
                label_wag = "Wage"

                df[label] = df[label_wag] + df[label_gen] + df[label_com]

                col_1 = df["Total_Reward_" + str(choice)].loc[:, cond]
                col_2 = df[label].loc[:, cond]

                np.testing.assert_array_almost_equal(col_1, col_2)

            for choice in [3, 4]:
                label = "Ex_Post_Reward_{}".format(choice)
                label_sys = "Systematic_Reward_{}".format(choice)
                label_sho = "Shock_Reward_{}".format(choice)

                df[label] = df[label_sys] * df[label_sho]
                df[label] = df[label_sys] + df[label_sho]

                # The equality does not hold if a state is inadmissible.
                cond = df["Years_Schooling"] != edu_spec["max"]

                col_1 = df["Total_Reward_" + str(choice)].loc[:, cond]
                col_2 = df[label].loc[:, cond]

                np.testing.assert_array_almost_equal(col_1, col_2)

        # If the model is deterministic, all shocks should be equal to zero. Of course,
        # one after exponentiation for wages.
        if is_deterministic:
            for i in range(1, 5):
                label = "Shock_Reward_{}".format(i)
                if i in [1, 2]:
                    cond = df[label] == 1
                else:
                    cond = df[label] == 0
                assert np.all(cond)

    @pytest.mark.skip(
        "get_total_values was removed. We can rewrite the test with the similar"
        "get_continuation_value function. For that, we need to be able to inject "
        "Fortran outputs into the python function. As a new PR will redefine the state"
        "space object, we will tackle the test afterwards."
    )
    def test_4(self):
        """ Testing the return values for the total values in case of myopic
        individuals.

        """
        constr = {"edu_spec": {"max": 99}}
        params_spec, options_spec = generate_random_model(
            myopic=True, point_constr=constr
        )

        # The equality below does not hold if schooling is an inadmissible state.

        respy_obj = RespyCls(params_spec, options_spec)
        respy_obj, _ = respy_obj.simulate()

        (
            num_periods,
            optim_paras,
            edu_spec,
            mapping_state_idx,
            periods_emax,
            states_all,
            periods_rewards_systematic,
            states_number_period,
        ) = dist_class_attributes(
            respy_obj,
            "num_periods",
            "optim_paras",
            "edu_spec",
            "mapping_state_idx",
            "periods_emax",
            "states_all",
            "periods_rewards_systematic",
            "states_number_period",
        )

        period = np.random.choice(range(num_periods))
        k = np.random.choice(range(states_number_period[period]))

        rewards_systematic = periods_rewards_systematic[period, k, :]
        draws = np.random.normal(size=4)

        total_values, rewards_ex_post = get_total_values(
            period,
            num_periods,
            optim_paras,
            rewards_systematic,
            draws,
            edu_spec,
            mapping_state_idx,
            periods_emax,
            k,
            states_all,
        )

        np.testing.assert_almost_equal(total_values, rewards_ex_post)
