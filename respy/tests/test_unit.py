import numpy as np
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import distribute_parameters
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.pre_processing.model_processing import write_init_file
import pytest

from respy.tests.codes.random_init import generate_init
from respy import RespyCls


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

            # Generate random request
            generate_init()

            # Process request and write out again.
            respy_obj = RespyCls("test.respy.ini")
            respy_obj.write_out("alt.respy.ini")

            # Read both initialization files and compare
            base_ini = open("test.respy.ini", "r").read()
            alt_ini = open("alt.respy.ini", "r").read()
            assert base_ini == alt_ini

    def test_3(self):
        """ Testing some of the relationships in the simulated dataset.
        """
        # Generate constraint periods
        constr = dict()
        constr["flag_deterministic"] = np.random.choice([True, False])
        constr["flag_myopic"] = np.random.choice([True, False])

        # Generate random initialization file
        generate_init(constr)

        # Perform toolbox actions
        respy_obj = RespyCls("test.respy.ini")
        _, df = respy_obj.simulate()

        # Check special case
        optim_paras, num_types, edu_spec, num_periods = dist_class_attributes(
            respy_obj, "optim_paras", "num_types", "edu_spec", "num_periods"
        )

        shocks_cholesky = optim_paras["shocks_cholesky"]

        is_deterministic = np.count_nonzero(shocks_cholesky) == 0

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
        if respy_obj.get_attr("is_myopic"):
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
        constr = dict()
        constr["flag_myopic"] = True

        init_dict = generate_init(constr)

        # The equality below does not hold if schooling is an inadmissible state.
        init_dict["EDUCATION"]["max"] = 99
        write_init_file(init_dict)

        respy_obj = RespyCls("test.respy.ini")

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
