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
from respy.python.solve.solve_auxiliary import StateSpace
from respy.python.shared.shared_auxiliary import (
    get_continuation_value_and_ex_post_rewards,
)
from respy.python.evaluate.evaluate_python import create_draws_and_prob_wages


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

    def test_4(self):
        """ Testing the return values for the total values in case of myopic
        individuals for one period.

        Note
        ----
        The original test was designed to use Fortran rewards and calculate the total
        values and rewards ex post in Python and see whether they match. As both
        versions diverged in their implementation, we will implement the test with the
        Python version and check the equality of Fortran and Python outputs at all
        stages.

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
            num_types,
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
            "num_types",
            "optim_paras",
            "edu_spec",
            "mapping_state_idx",
            "periods_emax",
            "states_all",
            "periods_rewards_systematic",
            "states_number_period",
        )

        # We have to create the state space and calculate the rewards in the Python
        # version as we later need the wages which are not part of
        # ``periods_rewards_systematic``.
        state_space = StateSpace(
            num_periods,
            num_types,
            edu_spec["start"],
            edu_spec["max"],
            optim_paras,
        )

        # Check that rewards match
        _, _, pyth, _ = state_space._get_fortran_counterparts()
        np.testing.assert_almost_equal(
            pyth, periods_rewards_systematic, decimal=15
        )

        period = np.random.choice(num_periods)
        draws = np.random.normal(size=4)

        # Internalize periods_emax
        state_space._create_attributes_from_fortran_counterparts(periods_emax)

        # Unpack necessary attributes
        rewards_period = state_space.get_attribute_from_period(
            "rewards", period
        )
        emaxs_period = state_space.get_attribute_from_period("emaxs", period)[
            :, :4
        ]

        total_values, rewards_ex_post = get_continuation_value_and_ex_post_rewards(
            rewards_period[:, -2:],
            rewards_period[:, :4],
            emaxs_period,
            draws.reshape(1, -1),
            optim_paras["delta"],
        )

        np.testing.assert_equal(total_values, rewards_ex_post)


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
