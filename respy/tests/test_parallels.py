# standard library
import numpy as np
import pytest

# testing library
from codes.random_init import generate_random_dict

# project library
from respy.python.shared.shared_auxiliary import print_init_dict
from respy.evaluate import evaluate

from respy import simulate
from respy import RespyCls


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
        init_dict = generate_random_dict(constr)

        base = None
        for is_parallel in [True, False]:

            init_dict['PROGRAM']['parallelism'] = is_parallel
            print_init_dict(init_dict)

            respy_obj = RespyCls('test.respy.ini')
            respy_obj = simulate(respy_obj)
            crit_val = evaluate(respy_obj)

            if base is None:
                base = crit_val
            np.testing.assert_equal(base, crit_val)

    def test_2(self):

        pass
        # Testing parallel vs scalar functions
        #num_slaves = np.random.randint(1, 5)
        #cmd = 'mpiexec /home/peisenha/restudToolbox/package/respy/fortran/bin' \
        #      '/testing_parallel_scalar ' + str(num_slaves)
        #os.system(cmd)
        #assert not os.path.exists('.error.testing')
