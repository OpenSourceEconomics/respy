# standard library
import numpy as np

import pytest

# testing library
from codes.random_init import generate_random_dict

# project library
from respy.python.shared.shared_auxiliary import print_init_dict
from respy.python.shared.shared_constants import IS_PARALLEL

from respy import estimate
from respy import simulate
from respy import RespyCls


@pytest.mark.skipif(not IS_PARALLEL, reason='No PARALLELISM available')
@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ This test ensures that it makes no difference whether the
        criterion function is evaluated in parallel or not.
        """
        # Generate random initialization file
        constr = dict()
        constr['version'] = 'FORTRAN'
        constr['maxfun'] = np.random.randint(0, 10)
        init_dict = generate_random_dict(constr)

        base = None
        for is_parallel in [True, False]:

            init_dict['PARALLELISM']['flag'] = is_parallel
            print_init_dict(init_dict)

            respy_obj = RespyCls('test.respy.ini')
            respy_obj = simulate(respy_obj)
            _, crit_val = estimate(respy_obj)

            if base is None:
                base = crit_val
            np.testing.assert_almost_equal(base, crit_val)

    def test_2(self):
        """ This test ensures that the record files are identical.
        """
        # Generate random initialization file. The number of periods is
        # higher than usual as only FORTRAN implementations are used to
        # solve the random request. This ensures that also some cases of
        # interpolation are explored.
        constr = dict()
        constr['version'] = 'FORTRAN'
        constr['periods'] = np.random.randint(3, 10)

        init_dict = generate_random_dict(constr)

        base_log = None
        for is_parallel in [False, True]:

            init_dict['PARALLELISM']['flag'] = is_parallel
            print_init_dict(init_dict)

            respy_obj = RespyCls('test.respy.ini')
            simulate(respy_obj)

            # Check for identical record
            if base_log is None:
                base_log = open('sol.respy.log', 'r').read()
            assert open('sol.respy.log', 'r').read() == base_log
