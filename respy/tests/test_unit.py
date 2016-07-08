import numpy as np
import pytest

from codes.random_init import generate_init

from respy.python.estimate.estimate_auxiliary import get_optim_paras
from respy.python.shared.shared_auxiliary import dist_optim_paras
from respy import solve
from respy import RespyCls
from respy import simulate


@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """

    def test_1(self):
        """ Testing ten admissible realizations of state space for the first
        three periods.
        """
        # Generate constraint periods
        constraints = dict()
        constraints['periods'] = np.random.randint(3, 5)

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')

        respy_obj = solve(respy_obj)

        simulate(respy_obj)

        # Distribute class attributes
        states_number_period = respy_obj.get_attr('states_number_period')

        states_all = respy_obj.get_attr('states_all')

        # The next hard-coded results assume that at least two more
        # years of education are admissible.
        edu_max = respy_obj.get_attr('edu_max')
        edu_start = respy_obj.get_attr('edu_start')

        if edu_max - edu_start < 2:
            return

        # The number of admissible states in the first three periods
        for j, number_period in enumerate([1, 4, 13]):
            assert (states_number_period[j] == number_period)

        # The actual realizations of admissible states in period one
        assert ((states_all[0, 0, :] == [0, 0, 0, 1]).all())

        # The actual realizations of admissible states in period two
        states = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]]
        states += [[1, 0, 0, 0]]

        for j, state in enumerate(states):
            assert ((states_all[1, j, :] == state).all())

        # The actual realizations of admissible states in period three
        states = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
        states += [[0, 0, 2, 1], [0, 1, 0, 0], [0, 1, 1, 0]]
        states += [[0, 1, 1, 1], [0, 2, 0, 0], [1, 0, 0, 0]]
        states += [[1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0]]
        states += [[2, 0, 0, 0]]

        for j, state in enumerate(states):
            assert ((states_all[2, j, :] == state).all())

    def test_2(self):
        """ Testing whether back-and-forth transformation have no effect.
        """
        # Generate random request
        paras_fixed = np.random.choice([True, False], 26)

        for i in range(10):
            # Create random parameter vector
            base = np.random.uniform(size=26)
            x = base.copy()

            # Apply numerous transformations
            for j in range(10):
                args = dist_optim_paras(x, is_debug=True)
                args += ('all', paras_fixed)
                x = get_optim_paras(*args, is_debug=True)

            # Checks
            np.testing.assert_allclose(base, x)

    def test_3(self):
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
