import numpy as np
import pytest

from respy.python.shared.shared_auxiliary import print_init_dict
from respy.python.shared.shared_constants import IS_FORTRAN
from codes.auxiliary import write_interpolation_grid
from codes.random_init import generate_init
from codes.auxiliary import write_draws

from respy import estimate
from respy import simulate
from respy import RespyCls


@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ This test ensures that using the ambiguity functionality with a
        level of zero yields the same results as using the risk functionality
        directly.

        """
        max_draws = np.random.randint(10, 100)

        constr = dict()
        constr['flag_parallelism'] = False
        constr['max_draws'] = max_draws
        constr['level'] = 0.01
        constr['maxfun'] = 0

        init_dict = generate_init(constr)

        # We also check explicitly across the different program implementations.
        num_periods = init_dict['BASICS']['periods']
        write_draws(num_periods, max_draws)
        write_interpolation_grid('test.respy.ini')

        versions = ['PYTHON']
        if IS_FORTRAN:
            versions += ['FORTRAN']

        base_val = None
        for version in versions:
            for is_ambiguity in [True, False]:

                init_dict['PROGRAM']['version'] = version

                print_init_dict(init_dict)

                respy_obj = RespyCls('test.respy.ini')

                simulate(respy_obj)
                _, crit_val = estimate(respy_obj)

                if base_val is None:
                    base_val = crit_val

                np.testing.assert_allclose(base_val, crit_val)

    def test_2(self):
        """ This test ensures that it does not matter which version runs
        the ambiguity codes.
        """

        max_draws = np.random.randint(10, 100)

        constr = dict()
        constr['flag_parallelism'] = False
        constr['max_draws'] = max_draws
        constr['level'] = np.random.uniform()
        constr['maxfun'] = 0

        init_dict = generate_init(constr)

        num_periods = init_dict['BASICS']['periods']
        write_draws(num_periods, max_draws)
        write_interpolation_grid('test.respy.ini')

        versions = ['PYTHON']
        if IS_FORTRAN:
            versions += ['FORTRAN']

        base_val = None
        for version in versions:
            print(version)
            init_dict['PROGRAM']['version'] = version

            print_init_dict(init_dict)

            respy_obj = RespyCls('test.respy.ini')

            simulate(respy_obj)
            _, crit_val = estimate(respy_obj)

            if base_val is None:
                base_val = crit_val

            np.testing.assert_allclose(base_val, crit_val)
