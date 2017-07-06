import numpy as np
import pytest

from respy.python.shared.shared_auxiliary import back_out_systematic_wages
from respy.python.solve.solve_auxiliary import calculate_systematic_wages
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.solve.solve_auxiliary import construct_covariates
from respy.python.shared.shared_auxiliary import dist_optim_paras
from respy.python.shared.shared_auxiliary import get_optim_paras
from codes.random_init import generate_init
from respy import simulate
from respy import RespyCls


@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ Testing whether back-and-forth transformation have no effect.
        """
        for i in range(10):
            num_types = np.random.randint(1, 5)
            num_paras = 50 + (num_types - 1) * 6

            # Create random parameter vector
            base = np.random.uniform(size=num_paras)

            x = base.copy()

            # Apply numerous transformations
            for j in range(10):
                optim_paras = dist_optim_paras(x, is_debug=True)
                args = (optim_paras, num_paras, 'all', True)
                x = get_optim_paras(*args)

            # Checks
            np.testing.assert_allclose(base, x)

    def test_2(self):
        """ Testing whether the back and forth transformation works.
        """
        for _ in range(100):

            # Generate random request
            generate_init()

            # Process request and write out again.
            respy_obj = RespyCls('test.respy.ini')
            respy_obj.write_out('alt.respy.ini')

            # Read both initialization files and compare
            base_ini = open('test.respy.ini', 'r').read()
            alt_ini = open('alt.respy.ini', 'r').read()
            assert base_ini == alt_ini

    def test_3(self):
        """ Testing some of the relationships in the simulated dataset.
        """
        # Generate constraint periods
        constr = dict()
        constr['flag_deterministic'] = np.random.choice([True, False])
        constr['flag_myopic'] = np.random.choice([True, False])

        # Generate random initialization file
        generate_init(constr)

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')
        _, df = simulate(respy_obj)

        # Check special case
        optim_paras, num_types, edu_spec, num_periods = dist_class_attributes(respy_obj,
            'optim_paras', 'num_types', 'edu_spec', 'num_periods')

        shocks_cholesky = optim_paras['shocks_cholesky']

        is_deterministic = (np.count_nonzero(shocks_cholesky) == 0)

        # The systematic component for the alternative to stay home should always be identical
        # by type, at least within periods.
        for period in range(num_periods):
            assert (df['Systematic_Reward_4'].loc[:, period].nunique() <= num_types)

        # We can back out the wage information from other information provided in the simulated
        # dataset.
        for choice in [1, 2]:
            cond = (df['Choice'] == choice)
            label_sys = 'Systematic_Reward_{}'.format(choice)
            label_sho = 'Shock_Reward_{}'.format(choice)
            label_gen = 'General_Reward_{}'.format(choice)
            label_com = 'Common_Reward'
            df['Ex_Post_Reward'] = (df[label_sys] - df[label_gen] - df[label_com]) * df[label_sho]

            col_1 = df['Ex_Post_Reward'].loc[:, cond]
            col_2 = df['Wage'].loc[:, cond]
            np.testing.assert_array_almost_equal(col_1, col_2)

        # In the myopic case, the total reward should the equal to the ex post rewards.
        if respy_obj.get_attr('is_myopic'):
            # The shock only affects the skill-function and not the other components determining
            # the overall reward.
            for choice in [1, 2]:
                cond = (df['Choice'] == choice)

                label = 'Ex_Post_Reward_{}'.format(choice)
                label_gen = 'General_Reward_{}'.format(choice)
                label_com = 'Common_Reward'
                label_wag = 'Wage_Internal'

                df[label] = df[label_wag] + df[label_gen] + df[label_com]

                col_1 = df['Total_Reward_' + str(choice)].loc[:, cond]
                col_2 = df[label].loc[:, cond]

                np.testing.assert_array_almost_equal(col_1, col_2)

            for choice in [3, 4]:
                label = 'Ex_Post_Reward_{}'.format(choice)
                label_sys = 'Systematic_Reward_{}'.format(choice)
                label_sho = 'Shock_Reward_{}'.format(choice)

                df[label] = df[label_sys] * df[label_sho]
                df[label] = df[label_sys] + df[label_sho]

                # The equality does not hold if a state is inadmissible.
                cond = (df['Years_Schooling'] != edu_spec['max'])

                col_1 = df['Total_Reward_' + str(choice)].loc[:, cond]
                col_2 = df[label].loc[:, cond]

                np.testing.assert_array_almost_equal(col_1, col_2)

        # If the model is deterministic, all shocks should be equal to zero. Of course,
        # one after exponentiation for wages.
        if is_deterministic:
            for i in range(1, 5):
                label = 'Shock_Reward_{}'.format(i)
                if i in [1, 2]:
                    cond = (df[label] == 1)
                else:
                    cond = (df[label] == 0)
                assert np.all(cond)

    def test_4(self):
        """ Testing whether back and forth transformations for the wage does work.
        """
        # Generate random initialization file
        generate_init()

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')
        respy_obj, _ = simulate(respy_obj)

        periods_rewards_systematic, states_number_period, states_all, num_periods, optim_paras = \
        dist_class_attributes(respy_obj, 'periods_rewards_systematic', 'states_number_period',
            'states_all', 'num_periods', 'optim_paras')

        for _ in range(10):
            # Construct a random state for the calculations.
            period = np.random.choice(range(num_periods))
            k = np.random.choice(range(states_number_period[period]))

            rewards_systematic = periods_rewards_systematic[period, k, :]
            exp_a, exp_b, edu, activity_lagged, type_ = states_all[period, k, :]

            covariates = construct_covariates(exp_a, exp_b, edu, activity_lagged, type_, period)
            wages = calculate_systematic_wages(covariates, optim_paras)

            args = (rewards_systematic, exp_a, exp_b, edu, activity_lagged, optim_paras)
            rslt = back_out_systematic_wages(*args)

            np.testing.assert_almost_equal(rslt, wages)